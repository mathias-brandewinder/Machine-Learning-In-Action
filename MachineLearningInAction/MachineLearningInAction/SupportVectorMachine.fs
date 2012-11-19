namespace MachineLearning

module SupportVectorMachine =

    // an observation, its label, and current alpha estimate
    type Row = { Data: float list; Label: float; Alpha: float }
    // SVM algorithm input parameters
    type Parameters = { Tolerance: float; C: float }

    type Attempt<'a> = Success of 'a | Failure

    // Product of vectors
    let dot (vec1: float list) 
            (vec2: float list) =
        List.fold2 (fun acc v1 v2 -> 
            acc + v1 * v2) 0.0 vec1 vec2

    // Clip a value x that is out of the min/max bounds
    let clip (min, max) x =
        if (x > max)
        then max
        elif (x < min)
        then min
        else x