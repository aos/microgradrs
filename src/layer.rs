use crate::neuron::Neuron;
use crate::value::Value;

#[derive(Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        Layer {
            neurons: (0..nout).map(|_| Neuron::new(nin)).collect(),
        }
    }

    pub fn call(&self, x: Vec<Value>) -> Vec<Value> {
        self.neurons
            .iter()
            .map(|n| n.call(x.clone()))
            .collect::<Vec<Value>>()
    }

    pub fn neurons(&self) -> &Vec<Neuron> {
        &self.neurons
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}
