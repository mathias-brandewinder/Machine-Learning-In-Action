#load "AdaBoost.fs"
open MachineLearning.AdaBoost

open System

let dataset, classLabels = 
    [| [| 1.0; 2.1 |];
       [| 2.0; 1.1 |];
       [| 1.3; 1.0 |];
       [| 1.0; 1.0 |];
       [| 2.0; 1.0 |] |], 
    [| 1.0; 1.0; -1.0; -1.0; 1.0 |]

let stumpClassify dimension threshold (observation: float []) =
    if observation.[dimension] >= threshold // need to pass op: > and < are valid
    then 1.0
    else -1.0

let weightedError (ex: Example) weight classifier =
    if classifier(ex.Observation) = ex.Label then 0.0 else weight

let buildStump (sample: Example []) weights =
    seq {
        let numSteps = 10.0
        let dimensions = dataset.[0].Length
        for dim in 0 .. dimensions - 1 do
            let column = dataset |> Array.map(fun obs -> obs.[dim])
            let min, max = Array.min column, Array.max column
            let stepSize = (max - min) / numSteps
            for threshold in min .. stepSize .. max do
                let stump = stumpClassify dim threshold
                let error =
                    Array.map2 (fun example w -> 
                        weightedError example w stump) sample weights
                    |> Array.sum
                yield dim, threshold, error }
    |> Seq.minBy (fun (dim, thresh, err) -> err)

let classify model observation = 
    let aggregate = List.sumBy (fun st -> 
        st.Alpha * st.Classifier observation) model
    match aggregate > 0.0 with 
    | true  ->  1.0
    | false -> -1.0

let train dataset labels iterations errorLimit =
    // Prepare data
    let sample = Array.map2 (fun obs lbl -> 
        { Observation = obs; Label = lbl } ) dataset labels

    // Recursively create new stumps and observation weights
    let rec update iter stumps weights =
        // Create best classifier given current weights
        let dim, thresh, err = buildStump sample weights
        printfn "Error: %f" err
        let stump = stumpClassify dim thresh
        let alpha = 0.5 * log ((1.0 - err) / err)
        let learner = { Alpha = alpha; Classifier = stump }

        // Update weights based on new classifier performance
        let weights' = 
            Array.map2 (fun obs weight -> 
                match stump obs.Observation = obs.Label with
                | true  -> weight * exp (-alpha)
                | false -> weight * exp alpha) sample weights
            |> normalize

        // Append new stump to the stumps list
        let stumps' = learner :: stumps

        // Search termination
        match iter < iterations with
        | false -> stumps'
        | true  ->
            // compute aggregate error
            let errorRate = Array.averageBy (fun obs -> 
                if ((classify stumps' obs.Observation) = obs.Label) then 0.0 else 1.0) sample
            printfn "Error rate: %f" errorRate
            match errorRate > errorLimit with
            | false -> stumps'
            | true  -> update (iter + 1) stumps' weights'

    let size = Array.length dataset
    let weights = [| for i in 1 .. size -> 1.0 / (float)size |]

    update 0 [] weights

let model = train dataset classLabels 5 0.05
let classifier = classify model
Array.zip dataset classLabels 
|> Array.iter (fun (d, l) -> printfn "Real %f Pred %f" l (classifier d)) 