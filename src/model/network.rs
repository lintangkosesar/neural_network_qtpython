use ndarray::{Array2, Axis};
use ndarray_rand::RandomExt;
use serde::{Serialize, Deserialize};

use crate::model::layers::{relu, relu_derivative, softmax};
use crate::data::preprocessing::DataStats;

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

#[derive(Debug)]
pub struct PredictionResult {
    pub class: i32,
    pub probabilities: Vec<f64>,
}

impl PredictionResult {
    pub fn display(&self) {
        let category = match self.class {
            0 => "BAIK",
            1 => "SEDANG",
            2 => "TIDAK SEHAT",
            _ => "UNKNOWN"
        };
        
        println!("\nPrediction Results:");
        println!("- BAIK: {:.2}%", self.probabilities[0] * 100.0);
        println!("- SEDANG: {:.2}%", self.probabilities[1] * 100.0);
        println!("- TIDAK SEHAT: {:.2}%", self.probabilities[2] * 100.0);
        println!("\nPredicted air quality category: {}", category);
    }
}

impl NeuralNetwork {
    pub fn new(
        input_size: usize, 
        hidden_size1: usize, 
        hidden_size2: usize, 
        hidden_size3: usize, 
        output_size: usize
    ) -> Self {
        let he_init = |size: usize| (2.0 / size as f64).sqrt();
        let weights1 = Array2::random(
            (input_size, hidden_size1), 
            ndarray_rand::rand_distr::Uniform::new(-he_init(input_size), he_init(input_size))
        );
        let bias1 = Array2::zeros((1, hidden_size1));
        let weights2 = Array2::random(
            (hidden_size1, hidden_size2), 
            ndarray_rand::rand_distr::Uniform::new(-he_init(hidden_size1), he_init(hidden_size1))
        );
        let bias2 = Array2::zeros((1, hidden_size2));
        let weights3 = Array2::random(
            (hidden_size2, hidden_size3), 
            ndarray_rand::rand_distr::Uniform::new(-he_init(hidden_size2), he_init(hidden_size2))
        );
        let bias3 = Array2::zeros((1, hidden_size3));
        let weights4 = Array2::random(
            (hidden_size3, output_size), 
            ndarray_rand::rand_distr::Uniform::new(-he_init(hidden_size3), he_init(hidden_size3))
        );
        let bias4 = Array2::zeros((1, output_size));
        
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

    pub fn forward(&self, x: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
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

    pub fn train(&mut self, x: &Array2<f64>, y: &Array2<f64>, learning_rate: f64, lambda: f64) {
        let (hidden_output1, hidden_output2, hidden_output3, output_output) = self.forward(x);

        // Backpropagation
        let output_error = &output_output - y;
        let output_delta = output_error;

        let hidden_error3 = output_delta.dot(&self.weights4.t());
        let hidden_delta3 = hidden_error3 * relu_derivative(&hidden_output3);

        let hidden_error2 = hidden_delta3.dot(&self.weights3.t());
        let hidden_delta2 = hidden_error2 * relu_derivative(&hidden_output2);

        let hidden_error1 = hidden_delta2.dot(&self.weights2.t());
        let hidden_delta1 = hidden_error1 * relu_derivative(&hidden_output1);

        // Update weights and biases
        self.weights4 -= &(learning_rate * (hidden_output3.t().dot(&output_delta) + lambda * &self.weights4));
        self.bias4 -= &(learning_rate * output_delta.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0)));

        self.weights3 -= &(learning_rate * (hidden_output2.t().dot(&hidden_delta3) + lambda * &self.weights3));
        self.bias3 -= &(learning_rate * hidden_delta3.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0)));

        self.weights2 -= &(learning_rate * (hidden_output1.t().dot(&hidden_delta2) + lambda * &self.weights2));
        self.bias2 -= &(learning_rate * hidden_delta2.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0)));

        self.weights1 -= &(learning_rate * (x.t().dot(&hidden_delta1) + lambda * &self.weights1));
        self.bias1 -= &(learning_rate * hidden_delta1.sum_axis(ndarray::Axis(0)).insert_axis(ndarray::Axis(0)));
    }

    pub fn predict(&self, input: &[f64], stats: &DataStats) -> PredictionResult {
        // Ubah slice input menjadi Array2<f64>
        let input = Array2::from_shape_vec((1, input.len()), input.to_vec()).unwrap();
    
        // Normalisasi input
        let mean = stats.mean.view().insert_axis(ndarray::Axis(0));
        let std = stats.std.view().insert_axis(ndarray::Axis(0));
        let input_normalized = (&input - &mean) / &std;
    
        // Forward pass
        let (_, _, _, output) = self.forward(&input_normalized);
    
        // Ambil hasil prediksi dan probabilitas
        let probabilities = output.row(0).to_vec();
        let class = probabilities.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index as i32)
            .unwrap_or(-1);
    
        PredictionResult {
            class,
            probabilities,
        }
    }

    pub fn loss(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
        -(y_true * y_pred.mapv(f64::ln)).sum()
    }

    pub fn accuracy(&self, y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
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
    
}