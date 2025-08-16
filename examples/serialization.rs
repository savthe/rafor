use std::fs::File;
use rafor::rf::Classifier;

fn main() {
    let dataset = [0.7, 0.0, 0.8, 1.0, 0.7, 0.0];
    let targets = [1, 5, 1];
    let predictor = Classifier::fit(&dataset, &targets, &Classifier::train_defaults());

    // Storing model.
    let mut fout = File::create("model.bin").unwrap();
    let config = bincode::config::standard();
    // Requires dependency bincode with 'serde' feature in Cargo.toml:
    // bincode = { version = "2.0", features = ["serde"] }
    let _ = bincode::serde::encode_into_std_write(&predictor, &mut fout, config);

    // Loading stored model.
    let mut fin = File::open("model.bin").unwrap();
    let config = bincode::config::standard();
    let predictor: Classifier = bincode::serde::decode_from_std_read(&mut fin, config).unwrap();

    let predictions = predictor.predict(&dataset, 1);
    assert_eq!(&predictions, &[1, 5, 1]);
}
