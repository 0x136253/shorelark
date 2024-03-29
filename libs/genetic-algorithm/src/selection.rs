use crate::individual::Individual;
use rand::{RngCore, prelude::SliceRandom};

pub trait SelectionMethod {
    fn select<'a, I>(
        &self,
        rng: &mut dyn RngCore,
        population: &'a [I],
    ) -> &'a I
    where
        I:Individual;
}

#[derive(Clone, Debug, Default)]
pub struct RouletteWheelSelection;

impl SelectionMethod for RouletteWheelSelection {
    fn select<'a, I>(
        &self,
        rng: &mut dyn RngCore,
        population: &'a [I],
    ) -> &'a I
    where
        I:Individual 
    {
        population
            .choose_weighted(rng, |individual| individual.fitness())
            .expect("got an empty population")
    }
}

#[cfg(test)]
mod test{
    use std::{vec, collections::BTreeMap};

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::individual::{TestIndividual, Individual};

    use super::{RouletteWheelSelection, SelectionMethod};


    #[test]
    fn test() {
        let method = RouletteWheelSelection::default();
        let mut rng = ChaCha8Rng::from_seed(Default::default());

        let population = vec![
            TestIndividual::new(2.0),
            TestIndividual::new(1.0),
            TestIndividual::new(4.0),
            TestIndividual::new(3.0),
        ];
    
        let actual_histogram: BTreeMap<i32, _> = (0..1000)
            .map(|_|method.select(&mut rng, &population))
            .fold(Default::default(), |mut histogram,individual|{
                *histogram
                    .entry(individual.fitness() as _)
                    .or_default() += 1;
                
                histogram
            });
        
        let expected_histogram = maplit::btreemap! {
            1=> 98,
            2=> 202,
            3=> 278,
            4=> 422,
        };

        assert_eq!(actual_histogram, expected_histogram);
    }
}