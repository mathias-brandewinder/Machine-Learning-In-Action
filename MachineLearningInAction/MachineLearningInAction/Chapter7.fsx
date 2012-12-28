open System

let dataset, classLabels = 
    [| [| 1.0; 2.1 |];
       [| 2.0; 1.1 |];
       [| 1.3; 1.0 |];
       [| 1.0; 1.0 |];
       [| 2.0; 1.0 |] |], 
    [| 1.0; 1.0; -1.0; -1.0; 1.0 |]

let stumpClassify (observation: float []) dimension threshold =
    if observation.[dimension] >= threshold
    then 1.0
    else -1.0

let weightedError (observation: float[]) label weight dimension threshold =
    let prediction = stumpClassify observation dimension threshold
    if prediction = label then 0.0 else weight

let buildStump (dataset:float[][]) classLabels weights  =
    seq {
        let numSteps = 10.0
        let dimensions = dataset.[0].Length
        for dim in 0 .. dimensions - 1 do
            let column = dataset |> Array.map(fun obs -> obs.[dim])
            let min, max = Array.min column, Array.max column
            let stepSize = (max - min) / numSteps
            for threshold in min .. stepSize .. max do
                let error =
                    Array.zip3 dataset classLabels weights
                    |> Array.map (fun (obs, lbl, w) -> 
                        weightedError obs lbl w dim threshold)
                    |> Array.sum
                yield dim, threshold, error }
    |> Seq.minBy (fun (dim, thresh, err) -> err)

let weights = [| 0.2; 0.2; 0.2; 0.2; 0.2 |]
buildStump dataset classLabels weights