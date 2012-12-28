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

let weightedError (observation: float[]) label weight classifier =
    let prediction = classifier observation
    if prediction = label then 0.0 else weight

let buildStump (dataset:float[][]) classLabels weights =
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
                    Array.zip3 dataset classLabels weights
                    |> Array.map (fun (obs, lbl, w) -> 
                        weightedError obs lbl w stump)
                    |> Array.sum
                yield dim, threshold, error }
    |> Seq.minBy (fun (dim, thresh, err) -> err)

let weights = [| 0.2; 0.2; 0.2; 0.2; 0.2 |]
buildStump dataset classLabels weights

type Observation = { Data: float []; Label: float }
type WeakLearner = { Alpha: float; Classifier: float [] -> float }

let normalize (weights: float []) = 
    let total = weights |> Array.sum 
    Array.map (fun w -> w / total) weights

let classify model observation = 
    let aggregate = List.sumBy (fun st -> 
        st.Alpha * st.Classifier observation) model
    match aggregate > 0.0 with 
    | true  ->  1.0
    | false -> -1.0

let train dataset labels iterations errorLimit =
    // Prepare data
    let sample = Array.map2 (fun data lbl -> 
        { Data = data; Label = lbl } ) dataset labels

    // Recursively create new stumps and observation weights
    let rec update iter stumps weights =
        // Create best classifier given current weights
        let dim, thresh, err = buildStump dataset labels weights
        printfn "Error: %f" err
        let stump = stumpClassify dim thresh
        let alpha = 0.5 * log ((1.0 - err) / err)
        let learner = { Alpha = alpha; Classifier = stump }

        // Update weights based on new classifier performance
        let weights' = 
            Array.map2 (fun obs weight -> 
                match stump obs.Data = obs.Label with
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
                if ((classify stumps' obs.Data) = obs.Label) then 0.0 else 1.0) sample
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