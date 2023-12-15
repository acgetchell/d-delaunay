use uuid::Uuid;

pub fn make_uuid() -> Uuid {
    Uuid::new_v4()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_uuid() {
        let uuid = make_uuid();
        println!("make_uuid = {:?}", uuid);
    }
}
