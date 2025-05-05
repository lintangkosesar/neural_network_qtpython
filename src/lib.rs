use ndarray::array;
use csv::Reader;
use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use serde::{Serialize, Deserialize};
use std::{error::Error, fs::File};
use plotters::prelude::*;
use libc;
use std::fmt;

pub type ProgressCallback = extern "C" fn(epoch: i32, accuracy: f64, loss: f64);

pub struct TrainingHistory {
    pub epochs: Vec<usize>,
    pub accuracies: Vec<f64>,
    pub losses: Vec<f64>,
}

impl Default for TrainingHistory {
    fn default() -> Self {
        TrainingHistory {
            epochs: Vec::new(),
            accuracies: Vec::new(),
            losses: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct TrainedModel {
    pub network: NeuralNetwork,
    pub x_mean: Array1<f64>,
    pub x_std: Array1<f64>,
    pub final_accuracy: f64,
}

#[derive(Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub weights1: Array2<f64>,
    pub bias1: Array2<f64>,
    pub weights2: Array2<f64>,
    pub bias2: Array2<f64>,
    pub weights3: Array2<f64>,
    pub bias3: Array2<f64>,
    pub weights4: Array2<f64>,
    pub bias4: Array2<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PredictionResult {
    pub predicted_class: i32,
    pub probabilities: Vec<f64>,
}

impl fmt::Debug for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NeuralNetwork {{\n  weights1: {:?}\n  bias1: {:?}\n  weights2: {:?}\n  bias2: {:?}\n  weights3: {:?}\n  bias3: {:?}\n  weights4: {:?}\n  bias4: {:?}\n}}",
            self.weights1, self.bias1, 
            self.weights2, self.bias2,
            self.weights3, self.bias3,
            self.weights4, self.bias4)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn train_model_with_progress(
    csv_path: *const libc::c_char,
    epochs: i32,
    plot_path: *const libc::c_char,
    model_path: *const libc::c_char,
    accuracy: *mut f64,
    callback: ProgressCallback,
) -> bool {
    unsafe {
        let csv_path_str = std::ffi::CStr::from_ptr(csv_path).to_str().unwrap();
        let plot_path_str = std::ffi::CStr::from_ptr(plot_path).to_str().unwrap();
        let model_path_str = std::ffi::CStr::from_ptr(model_path).to_str().unwrap();

        match train_network_with_progress(csv_path_str, epochs, plot_path_str, callback) {
            Ok(model) => {
                println!("Model training completed successfully");
                println!("Model weights: {:?}", model.network);
                println!("Normalization params - mean: {:?}, std: {:?}", model.x_mean, model.x_std);
                
                *accuracy = model.final_accuracy;
                if let Ok(model_data) = bincode::serialize(&model) {
                    std::fs::write(model_path_str, model_data).is_ok()
                } else {
                    false
                }
            }
            Err(e) => {
                println!("Training failed: {}", e);
                false
            }
        }
    }
}

fn train_network_with_progress(
    csv_path: &str, 
    epochs: i32, 
    plot_path: &str,
    callback: ProgressCallback,
) -> Result<TrainedModel, Box<dyn Error>> {
    // Read dataset
    let file = File::open(csv_path)?;
    let mut rdr = Reader::from_reader(file);
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    for result in rdr.records() {
        let record = result?;

        // Parse data
        let pm10: f64 = record[0].parse().unwrap_or(0.0);
        let so2: f64 = record[1].parse().unwrap_or(0.0);
        let co: f64 = record[2].parse().unwrap_or(0.0);
        let o3: f64 = record[3].parse().unwrap_or(0.0);
        let no2: f64 = record[4].parse().unwrap_or(0.0);

        let category = &record[5];
        inputs.push(vec![pm10, so2, co, o3, no2]);
        outputs.push(match category {
            "BAIK" => vec![1.0, 0.0, 0.0],
            "SEDANG" => vec![0.0, 1.0, 0.0],
            "TIDAK SEHAT" => vec![0.0, 0.0, 1.0],
            _ => vec![0.0, 0.0, 0.0]
        });
    }

    let x = Array2::from_shape_vec((inputs.len(), 5), inputs.concat())?;
    let y = Array2::from_shape_vec((outputs.len(), 3), outputs.concat())?;

    // Normalize input
    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let x_std = x.std_axis(Axis(0), 1.0);
    let x_normalized = (x - &x_mean) / &x_std;

    println!("Input data normalized successfully");
    println!("Mean: {:?}", x_mean);
    println!("Std: {:?}", x_std);

    // Initialize neural network
    let mut nn = NeuralNetwork::new(5, 10, 10, 10, 3);
    let mut history = TrainingHistory::default();

    // Training
    let initial_learning_rate = 0.001;
    let lambda = 0.01;
    for epoch in 0..epochs as usize {
        let learning_rate = initial_learning_rate * (1.0 / (1.0 + 0.1 * (epoch as f64)));
        nn.train(&x_normalized, &y, learning_rate, lambda);
        
        // Evaluation and callback
        if epoch % 10 == 0 {
            let (_, _, _, output) = nn.forward(&x_normalized);
            let loss = cross_entropy_loss(&y, &output);
            let accuracy = calculate_accuracy(&y, &output);
            
            history.epochs.push(epoch);
            history.accuracies.push(accuracy);
            history.losses.push(loss);
            
            // Call callback to update progress
            callback(epoch as i32, accuracy, loss);
        }
    }

    // Create plot
    create_plot(&history, plot_path)?;
    
    // Calculate final accuracy
    let (_, _, _, val_output) = nn.forward(&x_normalized);
    let final_accuracy = calculate_accuracy(&y, &val_output);

    Ok(TrainedModel {
        network: nn,
        x_mean,
        x_std,
        final_accuracy,
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn predict_air_quality(
    pm10: f64,
    so2: f64,
    co: f64,
    o3: f64,
    no2: f64,
    model_path: *const libc::c_char,
) -> *mut PredictionResult {
    unsafe {
        let model_path_str = std::ffi::CStr::from_ptr(model_path).to_str().unwrap();
        
        println!("Loading model from: {}", model_path_str);
        
        let model_data = match std::fs::read(model_path_str) {
            Ok(data) => data,
            Err(e) => {
                println!("Failed to read model file: {}", e);
                return std::ptr::null_mut();
            },
        };
        
        let trained_model: TrainedModel = match bincode::deserialize::<TrainedModel>(&model_data) {
            Ok(model) => {
                println!("Model loaded successfully");
                println!("Model weights: {:?}", model.network);
                println!("Normalization params - mean: {:?}, std: {:?}", model.x_mean, model.x_std);
                model
            },
            Err(e) => {
                println!("Failed to deserialize model: {}", e);
                return std::ptr::null_mut();
            },
        };

        let prediction = predict(
            pm10, so2, co, o3, no2, 
            &trained_model.network, 
            &trained_model.x_mean, 
            &trained_model.x_std
        );

        println!("Prediction result: {:?}", prediction);

        Box::into_raw(Box::new(prediction))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn free_prediction_result(result: *mut PredictionResult) {
    if !result.is_null() {
        unsafe {
            let _ = Box::from_raw(result);
        }
    }
}

fn predict(
    pm10: f64, 
    so2: f64, 
    co: f64, 
    o3: f64, 
    no2: f64, 
    nn: &NeuralNetwork, 
    x_mean: &Array1<f64>, 
    x_std: &Array1<f64>
) -> PredictionResult {
    let input = array![[pm10, so2, co, o3, no2]];
    println!("Raw input: {:?}", input);
    
    let input_normalized = (input - x_mean) / x_std;
    println!("Normalized input: {:?}", input_normalized);

    let (_, _, _, output) = nn.forward(&input_normalized);
    println!("Network output: {:?}", output);

    let probabilities = output.row(0).to_vec();
    let class = probabilities.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index as i32)
        .unwrap_or(-1);

    PredictionResult {
        predicted_class: class,
        probabilities,
    }
}

fn calculate_accuracy(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    let predictions = y_pred.map_axis(Axis(1), |row| {
        row.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    });
    let true_labels = y_true.map_axis(Axis(1), |row| {
        row.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap()
    });
    predictions.iter()
        .zip(true_labels.iter())
        .filter(|&(a, b)| a == b)
        .count() as f64 / predictions.len() as f64
}

fn create_plot(history: &TrainingHistory, path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_epoch = *history.epochs.last().unwrap_or(&1) as u32;
    let max_loss = history.losses.iter().cloned().fold(f64::NAN, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Training Progress", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0u32..max_epoch, 0f64..1f64)?;

    chart.configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_desc("Epoch")
        .y_desc("Value")
        .draw()?;

    chart.draw_series(LineSeries::new(
        history.epochs.iter().zip(history.accuracies.iter()).map(|(&x, &y)| (x as u32, y)),
        &RED,
    ))?.label("Accuracy")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    let loss_scale = 1.0 / max_loss;
    chart.draw_series(LineSeries::new(
        history.epochs.iter().zip(history.losses.iter()).map(|(&x, &y)| (x as u32, y * loss_scale)),
        &BLUE,
    ))?.label("Loss (scaled)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn relu(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| if v > 0.0 { v } else { 0.0 })
}

fn relu_derivative(x: &Array2<f64>) -> Array2<f64> {
    x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let max_x = x.fold_axis(Axis(1), f64::NEG_INFINITY, |&a, &b| a.max(b));
    let exp_x = (x - &max_x.insert_axis(Axis(1))).mapv(f64::exp);
    let sum_exp_x = exp_x.sum_axis(Axis(1)).insert_axis(Axis(1));
    exp_x / sum_exp_x
}

fn cross_entropy_loss(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    -(y_true * y_pred.mapv(f64::ln)).sum()
}

impl NeuralNetwork {
    fn new(
        input_size: usize, 
        hidden_size1: usize, 
        hidden_size2: usize, 
        hidden_size3: usize, 
        output_size: usize
    ) -> Self {
        let he_init = |size: usize| (2.0 / size as f64).sqrt();
        let weights1 = Array::random(
            (input_size, hidden_size1), 
            Uniform::new(-he_init(input_size), he_init(input_size))
        );
        let bias1 = Array::zeros((1, hidden_size1));
        let weights2 = Array::random(
            (hidden_size1, hidden_size2), 
            Uniform::new(-he_init(hidden_size1), he_init(hidden_size1))
        );
        let bias2 = Array::zeros((1, hidden_size2));
        let weights3 = Array::random(
            (hidden_size2, hidden_size3), 
            Uniform::new(-he_init(hidden_size2), he_init(hidden_size2))
        );
        let bias3 = Array::zeros((1, hidden_size3));
        let weights4 = Array::random(
            (hidden_size3, output_size), 
            Uniform::new(-he_init(hidden_size3), he_init(hidden_size3))
        );
        let bias4 = Array::zeros((1, output_size));
        
        NeuralNetwork {
            weights1,
            bias1,
            weights2,
            bias2,
            weights3,
            bias3,
            weights4,
            bias4,
        }
    }

    fn forward(&self, x: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
        let hidden_input1 = x.dot(&self.weights1) + &self.bias1;
        let hidden_output1 = relu(&hidden_input1);

        let hidden_input2 = hidden_output1.dot(&self.weights2) + &self.bias2;
        let hidden_output2 = relu(&hidden_input2);

        let hidden_input3 = hidden_output2.dot(&self.weights3) + &self.bias3;
        let hidden_output3 = relu(&hidden_input3);

        let output_input = hidden_output3.dot(&self.weights4) + &self.bias4;
        let output_output = softmax(&output_input);

        (hidden_output1, hidden_output2, hidden_output3, output_output)
    }

    fn train(&mut self, x: &Array2<f64>, y: &Array2<f64>, learning_rate: f64, lambda: f64) {
        let (hidden_output1, hidden_output2, hidden_output3, output_output) = self.forward(x);

        let output_error = &output_output - y;
        let output_delta = output_error;

        let hidden_error3 = output_delta.dot(&self.weights4.t());
        let hidden_delta3 = hidden_error3 * relu_derivative(&hidden_output3);

        let hidden_error2 = hidden_delta3.dot(&self.weights3.t());
        let hidden_delta2 = hidden_error2 * relu_derivative(&hidden_output2);

        let hidden_error1 = hidden_delta2.dot(&self.weights2.t());
        let hidden_delta1 = hidden_error1 * relu_derivative(&hidden_output1);

        self.weights4 -= &(learning_rate * (hidden_output3.t().dot(&output_delta) + lambda * &self.weights4));
        self.bias4 -= &(learning_rate * output_delta.sum_axis(Axis(0)).insert_axis(Axis(0)));

        self.weights3 -= &(learning_rate * (hidden_output2.t().dot(&hidden_delta3) + lambda * &self.weights3));
        self.bias3 -= &(learning_rate * hidden_delta3.sum_axis(Axis(0)).insert_axis(Axis(0)));

        self.weights2 -= &(learning_rate * (hidden_output1.t().dot(&hidden_delta2) + lambda * &self.weights2));
        self.bias2 -= &(learning_rate * hidden_delta2.sum_axis(Axis(0)).insert_axis(Axis(0)));

        self.weights1 -= &(learning_rate * (x.t().dot(&hidden_delta1) + lambda * &self.weights1));
        self.bias1 -= &(learning_rate * hidden_delta1.sum_axis(Axis(0)).insert_axis(Axis(0)));
    }
}