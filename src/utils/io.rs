use crate::training::trainer::TrainedModel;
use std::error::Error;

pub fn save_model(path: &str, model: &TrainedModel) -> Result<(), Box<dyn Error>> {
    let model_data = bincode::serialize(model)?;
    std::fs::write(path, model_data)?;
    Ok(())
}

pub fn load_model(path: &str) -> Result<TrainedModel, Box<dyn Error>> {
    let model_data = std::fs::read(path)?;
    let model = bincode::deserialize(&model_data)?;
    Ok(model)
}