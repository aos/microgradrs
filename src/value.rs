use std::cell::{Ref, RefCell};
use std::collections::HashSet;
use std::fmt;
use std::hash::Hash;
use std::iter::Sum;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Value(Rc<RefCell<ValueInternal>>);

impl Value {
    pub fn from<T>(t: T) -> Self
    where
        T: Into<Value>,
    {
        t.into()
    }

    fn new(t: ValueInternal) -> Self {
        Self(Rc::new(RefCell::new(t)))
    }

    pub fn with_label(self, label: &str) -> Self {
        self.0.borrow_mut().label = Some(label.to_string());
        self
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn pow(&self, other: &Value) -> Self {
        let result = self.0.borrow().data.powf(other.0.borrow().data);

        let prop_fn: PropagateFn = |value| {
            let mut base = value.previous[0].0.borrow_mut();
            let power = value.previous[1].0.borrow();
            base.gradient += power.data * base.data.powf(power.data - 1.0) * value.gradient;
        };

        Value::new(ValueInternal::new(
            result,
            None,
            Some("^".to_string()),
            vec![self.clone(), other.clone()],
            Some(prop_fn),
        ))
    }

    pub fn tanh(&self) -> Self {
        let result = self.data().tanh();

        let prop_fn: PropagateFn = |value| {
            let mut previous = value.previous[0].0.borrow_mut();
            previous.gradient += (1.0 - previous.data.powf(2.0)) * previous.gradient;
        };

        Value::new(ValueInternal::new(
            result,
            None,
            Some("tanh".to_string()),
            vec![self.clone()],
            Some(prop_fn),
        ))
    }

    pub fn exp(&self) -> Self {
        let result = self.data().exp();

        let prop_fn: PropagateFn = |value| {
            let mut previous = value.previous[0].0.borrow_mut();
            previous.gradient += previous.data;
        };

        Value::new(ValueInternal::new(
            result,
            None,
            Some("exp".to_string()),
            vec![self.clone()],
            Some(prop_fn),
        ))
    }

    pub fn backward(&mut self) {
        let mut visited: HashSet<Value> = HashSet::new();

        self.0.borrow_mut().gradient = 1.0;
        self.backward_internal(&mut visited, self);
    }

    fn backward_internal(&self, visited: &mut HashSet<Value>, value: &Value) {
        if !visited.contains(value) {
            visited.insert(value.clone());

            let borrowed_value = value.0.borrow();
            if let Some(prop_fn) = borrowed_value.propagate {
                prop_fn(&borrowed_value);
            }

            for child_id in &value.0.borrow().previous {
                self.backward_internal(visited, child_id);
            }
        }
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.borrow().hash(state);
    }
}

impl<T: Into<f64>> From<T> for Value {
    fn from(t: T) -> Value {
        Value::new(ValueInternal::new(t.into(), None, None, Vec::new(), None))
    }
}

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, other: Value) -> Self::Output {
        add(&self, &other)
    }
}

impl<'a, 'b> Add<&'b Value> for &'a Value {
    type Output = Value;

    fn add(self, other: &'b Value) -> Self::Output {
        add(self, other)
    }
}

fn add(a: &Value, b: &Value) -> Value {
    let result = a.data() + b.data();

    let prop_fn: PropagateFn = |value| {
        let mut first = value.previous[0].0.borrow_mut();
        let mut second = value.previous[1].0.borrow_mut();

        first.gradient += value.gradient;
        second.gradient += value.gradient;
    };

    Value::new(ValueInternal::new(
        result,
        None,
        Some("+".to_string()),
        vec![a.clone(), b.clone()],
        Some(prop_fn),
    ))
}

impl Mul<Value> for Value {
    type Output = Self;

    fn mul(self, other: Value) -> Self::Output {
        mul(&self, &other)
    }
}

impl<'a, 'b> Mul<&'b Value> for &'a Value {
    type Output = Value;

    fn mul(self, other: &'b Value) -> Self::Output {
        mul(self, other)
    }
}

fn mul(a: &Value, b: &Value) -> Value {
    let result = a.data() * b.data();

    let prop_fn: PropagateFn = |value| {
        let mut first = value.previous[0].0.borrow_mut();
        let mut second = value.previous[1].0.borrow_mut();

        first.gradient += second.data * value.gradient;
        second.gradient += first.data * value.gradient;
    };

    Value::new(ValueInternal::new(
        result,
        None,
        Some("*".to_string()),
        vec![a.clone(), b.clone()],
        Some(prop_fn),
    ))
}

impl Sub for Value {
    type Output = Self;

    fn sub(self, other: Value) -> Self::Output {
        add(&self, &(-other))
    }
}

impl Neg for Value {
    type Output = Self;

    fn neg(self) -> Self::Output {
        mul(&self, &Value::from(-1))
    }
}

impl Div for Value {
    type Output = Self;

    fn div(self, other: Value) -> Self::Output {
        mul(&self, &other.pow(&Value::from(-1)))
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Value::from(0.0), |a, b| a + b)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value({})", self.data())
    }
}

type PropagateFn = fn(value: &Ref<ValueInternal>);

#[derive(Debug)]
struct ValueInternal {
    data: f64,
    gradient: f64,
    label: Option<String>,
    operation: Option<String>,
    previous: Vec<Value>,
    propagate: Option<PropagateFn>,
}

impl ValueInternal {
    fn new(
        data: f64,
        label: Option<String>,
        op: Option<String>,
        prev: Vec<Value>,
        propagate: Option<PropagateFn>,
    ) -> Self {
        ValueInternal {
            data,
            gradient: 0.0,
            label,
            operation: op,
            previous: prev,
            propagate,
        }
    }
}

impl PartialEq for ValueInternal {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.gradient == other.gradient
            && self.label == other.label
            && self.operation == other.operation
            && self.previous == other.previous
    }
}

impl Eq for ValueInternal {}

impl Hash for ValueInternal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
        self.gradient.to_bits().hash(state);
        self.label.hash(state);
        self.operation.hash(state);
        self.previous.hash(state);
    }
}
