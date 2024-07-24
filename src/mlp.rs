use crate::layer::Layer;
use crate::value::Value;

#[derive(Clone)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: Vec<usize>) -> Self {
        let nouts_len = nouts.len();
        let mut sz = nouts;
        sz.insert(0, nin);

        MLP {
            layers: (0..nouts_len)
                .map(|i| Layer::new(sz[i], sz[i + 1]))
                .collect(),
        }
    }

    pub fn call(&self, x: Vec<Value>) -> Vec<Value> {
        let mut z = x;
        for layer in &self.layers {
            z = layer.call(z.clone());
        }
        z
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|n| n.parameters()).collect()
    }
}
