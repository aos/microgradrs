use crate::value::Value;
use rand::prelude::*;

#[derive(Debug)]
pub struct Neuron {
    w: Vec<Value>,
    b: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut z: Vec<Value> = Vec::new();

        for _ in 0..nin {
            z.push(Value::from(rng.gen_range(-1.0..1.0)));
        }

        Neuron {
            w: z,
            b: Value::from(rng.gen_range(-1.0..1.0)),
        }
    }

    pub fn call(&self, x: Vec<f64>) -> Value {
        (x.iter()
            .map(|n| Value::from(*n))
            .zip(&self.w)
            .map(|(x, w)| &x * w)
            .sum::<Value>()
            + self.b.clone())
        .tanh()
    }
}
