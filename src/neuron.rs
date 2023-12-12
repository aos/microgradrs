use crate::value::Value;
use rand::Rng;

#[derive(Clone)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng();

        Neuron {
            weights: (0..nin)
                .map(|_| Value::from(rng.gen_range(-1.0..1.0)))
                .collect(),
            bias: Value::from(rng.gen_range(-1.0..1.0)),
        }
    }

    pub fn call(&self, x: Vec<Value>) -> Value {
        (x.iter()
            .zip(&self.weights)
            .map(|(x, w)| x * w)
            .sum::<Value>()
            + self.bias.clone())
        .tanh()
    }
}
