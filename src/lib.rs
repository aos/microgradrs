use std::ops::{Add, Mul};
use std::fmt;

#[derive(PartialEq, Debug, Default)]
pub struct Value<'a> {
    data: f64,
    grad: f64,
    label: Option<&'a str>,
    _prev: Option<Vec<Value<'a>>>,
    _op: Option<&'a str>,
}

impl<'a> Value<'a> {
    pub fn new(data: f64, label: Option<&'a str>) -> Self {
        Self {
            data,
            label,
            ..Default::default()
        }
    }

    pub fn label(&mut self, label: &'a str) {
        self.label = Some(label);
    }
}

impl Add for Value<'_> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            data: self.data + other.data,
            _op: Some("+"),
            _prev: Some(vec![self, other]),
            ..Default::default()
        }
    }
}

impl Mul for Value<'_> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            data: self.data * other.data,
            _op: Some("*"),
            _prev: Some(vec![self, other]),
            ..Default::default()
        }
    }
}

impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={})", self.data)
    }
}
