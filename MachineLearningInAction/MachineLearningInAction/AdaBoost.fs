namespace MachineLearning

module AdaBoost =

    open System

    // A "known example": an observation and its known class label
    type Example = { Observation: float []; Label: float }
    // A "weak learner": a rudimentary classifier and its weight Alpha
    type WeakLearner = { Alpha: float; Classifier: float [] -> float }

    // "Stump": classify based on whether value of a single feature 
    // is greater/lower than threshold.
    let stumpClassify dimension threshold op (observation: float []) =
        if op observation.[dimension] threshold then 1.0 else -1.0

    let weightedError (ex: Example) weight classifier =
        if classifier(ex.Observation) = ex.Label then 0.0 else weight

    // Generate stump classifiers for each feature, varying the
    // threshold and the comparison, and pick the stump that
    // has the lowest weighted error.
    let bestStump (sample: Example []) weights numSteps =
        seq {
            let dimensions = sample.[0].Observation.Length
            for dim in 0 .. dimensions - 1 do
                let column = sample |> Array.map(fun obs -> obs.Observation.[dim])
                let min, max = Array.min column, Array.max column
                let stepSize = (max - min) / numSteps
                for threshold in min .. stepSize .. max do
                    for op in [ (<=); (>=) ] do
                        let stump = stumpClassify dim threshold op
                        let error =
                            Seq.map2 (fun example weight -> 
                                weightedError example weight stump) sample weights
                            |> Seq.sum
                        yield stump, error }
        |> Seq.minBy (fun (stump, err) -> err)
        
    // Normalize an array so that its elements sum to 1.
    let normalize (weights: float []) = 
        let total = weights |> Array.sum 
        Array.map (fun w -> w / total) weights

    // Classify an observation using a list of weak learners
    // and their weight Alpha: compute the alpha-weighted sum 
    // of the predictions of each learner, and decide based on sign. 
    let classify model observation = 
        let aggregate = List.sumBy (fun weakLearner -> 
            weakLearner.Alpha * weakLearner.Classifier observation) model
        match aggregate > 0.0 with 
        | true  ->  1.0
        | false -> -1.0
    
    // Compute proportion of Examples (sample) properly classified
    // using a model (a list of alpha-weighted weak learners)
    let aggregateError sample model = 
        Seq.averageBy (fun obs -> 
            if (classify model obs.Observation = obs.Label) then 0.0 else 1.0) sample

    let train dataset labels iterations numSteps targetError =
        // Prepare data
        let sample = Array.map2 (fun obs lbl -> 
            { Observation = obs; Label = lbl } ) dataset labels

        // Recursively create new stumps and observation weights
        let rec update iter stumps weights =

            // Create best classifier given current weights
            let stump, err = bestStump sample weights numSteps
            let alpha = 0.5 * log ((1.0 - err) / err)
            let learner = { Alpha = alpha; Classifier = stump }

            // Update weights based on new classifier performance
            let weights' = 
                Array.map2 (fun obs weight -> 
                    match stump(obs.Observation) = obs.Label with
                    | true  -> weight * exp (-alpha)
                    | false -> weight * exp alpha) sample weights
                |> normalize

            // Append new stump to the stumps list
            let stumps' = learner :: stumps

            // Search termination
            match iter >= iterations with
            | true  -> stumps' // done, we passed iterations limit
            | false ->
                // compute aggregate error
                let error = aggregateError sample stumps' 
                match error <= targetError with
                | true  -> stumps' // done, we reached error target
                | false -> update (iter + 1) stumps' weights'

        // Initiate recursive update and create classifier from stumps
        let size = Array.length dataset
        let weights = [| for i in 1 .. size -> 1.0 / (float)size |]

        let model = update 0 [] weights
        classify model