open System

let dataset, classLabels = 
    [| [| 1.0; 2.1 |];
       [| 2.0; 1.1 |];
       [| 1.3; 1.0 |];
       [| 1.0; 1.0 |];
       [| 2.0; 1.0 |] |], 
    [| 1.0; 1.0; -1.0; -1.0; 1.0 |]

let stumpClassify dimension threshold (observation: float []) =
    if observation.[dimension] >= threshold
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

type WeakLearner = { Alpha: float; Classifier: float [] -> float }

let normalize (weights: float []) = 
    let total = weights |> Array.sum 
    Array.map (fun w -> w / total) weights

let train dataset labels =

    let rec update iter stumps w =

        let dim, thresh, err = buildStump dataset labels w
        printfn "Error: %f" err
        let stump = stumpClassify dim thresh
        let alpha = 0.5 * log ((1.0 - err) / err)
        let learner = { Alpha = alpha; Classifier = stump }

        let w' = 
            Array.zip3 dataset labels w
            |> Array.map (fun (obs, lbl, w) -> 
                match stump obs = lbl with
                | true  -> w * exp (-alpha)
                | false -> w * exp alpha)
            |> normalize
        let stumps' = learner :: stumps

        match iter > 0 with
        | false -> stumps'
        | true  -> update (iter - 1) stumps' w'

    let size = Array.length dataset
    let weights = [| for i in 1 .. size -> 1.0 / (float)size |]

    update 10 [] weights