use crate::model::network::NeuralNetwork;
use crate::training::history::TrainingHistory;
use crate::data::preprocessing::{load_and_preprocess_data, DataStats};
use std::error::Error;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct TrainedModel {
    pub network: NeuralNetwork,
    pub stats: DataStats,
}

pub fn train_model(
    csv_path: &str,
    epochs: usize,
    plot_path: &str,
) -> Result<TrainedModel, Box<dyn Error>> {
    // Load and preprocess data
    let (x_normalized, y, stats) = load_and_preprocess_data(csv_path)?;

    // Initialize network
    let mut nn = NeuralNetwork::new(5, 10, 10, 10, 3);
    let mut history = TrainingHistory::default();

    // Training parameters
    let initial_learning_rate = 0.001;
    let lambda = 0.01;

    println!("Starting training with {} epochs...", epochs);

    // Training loop
    for epoch in 0..epochs {
        let learning_rate = initial_learning_rate * (1.0 / (1.0 + 0.1 * (epoch as f64)));
        
        // Forward and backward pass
        nn.train(&x_normalized, &y, learning_rate, lambda);
        
        // Evaluate every 10 epochs
        if epoch % 10 == 0 || epoch == epochs - 1 {
            let (_, _, _, output) = nn.forward(&x_normalized);
            let loss = nn.loss(&y, &output);
            let accuracy = nn.accuracy(&y, &output);
            
            history.record(epoch, accuracy, loss);
            
            if epoch % 100 == 0 || epoch == epochs - 1 {
                println!("Epoch {}/{} - loss: {:.4}, accuracy: {:.2}%", 
                    epoch, epochs, loss, accuracy * 100.0);
            }
        }
    }

    // Save training plot
    crate::utils::plot::create_plot(&history, plot_path)?;

    Ok(TrainedModel {
        network: nn,
        stats,
    })
}