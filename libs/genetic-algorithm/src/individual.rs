use crate::chromosome::Chromosome;

pub trait Individual {
    fn fitness(&self) -> f32;
    fn chromosome(&self) -> &Chromosome;
    fn create(chromosome:Chromosome) -> Self;
}


#[cfg(test)]
#[derive(Debug,Clone,PartialEq)]
pub enum TestIndividual {

    /// For tests that require access to chromosome
    WithChromosome { chromosome: Chromosome },
    
    /// For tests that don't require access to chromosome
    WithFitness { fitness: f32 },
}

#[cfg(test)]
impl TestIndividual {
    pub fn new(fitness:f32) -> Self {
        Self::WithFitness { fitness}
    }
}

#[cfg(test)]
impl Individual for TestIndividual {
    fn fitness(&self) -> f32 {
        match self {
            Self::WithChromosome { chromosome } => {
                chromosome.iter().sum()
            },
            Self::WithFitness { fitness } => *fitness,
        }
    }

    fn chromosome(&self) -> &Chromosome {
        match self {
            Self::WithChromosome { chromosome } => chromosome,

            Self::WithFitness { .. } => {
                panic!("not supported for TestIndividual::WithFitness")
            },
        } 
    }

    fn create(chromosome:Chromosome) -> Self {
        Self::WithChromosome { chromosome }
    }
}