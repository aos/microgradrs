use crate::neuron::Neuron;
use crate::value::Value;

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        let mut z: Vec<Neuron> = Vec::new();

        for _ in 0..nout {
            z.push(Neuron::new(nin));
        }

        Layer { neurons: z }
    }

    pub fn call(&self, x: Vec<f64>) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|n| n.call(x.clone()))
            .collect::<Vec<Value>>()
    }

    pub fn neurons(&self) -> &Vec<Neuron> {
        &self.neurons
    }
}
