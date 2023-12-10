use std::collections::HashSet;
use std::hash::Hash;
use std::ops::{Add, Deref, Mul};
use std::rc::Rc;
use std::cell::{Ref, RefCell};

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Value(Rc<RefCell<ValueInternal>>);

impl Value {
    pub fn from<T>(t: T) -> Self
    where T: Into<Value>,
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

    pub fn tanh(self) -> Self {
        let t = self.data().tanh();

        let prop_fn: PropagateFn = |value| {
            let mut previous = value.previous[0].0.borrow_mut();
            previous.gradient += (1.0 - value.data.powf(2.0)) * value.gradient;
        };

        Value::new(ValueInternal::new(
            t,
            None,
            Some("tanh".to_string()),
            vec![self.clone()],
            Some(prop_fn),
        ))
    }

    pub fn backward(&self) {
        let mut visited: HashSet<Value> = HashSet::new();

        self.0.borrow_mut().gradient = 1.0;
        self.backward_internal(&mut visited, self);
    }

    fn backward_internal(&self, visited: &mut HashSet<Value>, value: &Value) {
        if !visited.contains(&value) {
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
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        mul(&self, &other)
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

// impl Deref for Value {
//     type Target = Rc<RefCell<ValueInternal>>;
// 
//     fn deref(&self) -> &Self::Target {
//         &self.0
//     }
// }

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
        self.data == other.data &&
        self.gradient == other.gradient &&
        self.label == other.label &&
        self.operation == other.operation &&
        self.previous == other.previous
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
