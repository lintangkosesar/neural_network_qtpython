use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
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

impl TrainingHistory {
    pub fn record(&mut self, epoch: usize, accuracy: f64, loss: f64) {
        self.epochs.push(epoch);
        self.accuracies.push(accuracy);
        self.losses.push(loss);
    }
}