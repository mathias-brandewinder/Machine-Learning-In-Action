#load "AdaBoost.fs"
open MachineLearning.AdaBoost

open System

let stumpClassify dimension threshold op (observation: float []) =
    if op observation.[dimension] threshold // need to pass op: > and < are valid
    then 1.0
    else -1.0

let weightedError (ex: Example) weight classifier =
    if classifier(ex.Observation) = ex.Label then 0.0 else weight

let buildStump (sample: Example []) weights =
    seq {
        let numSteps = 10.0
        let dimensions = sample.[0].Observation.Length
        for dim in 0 .. dimensions - 1 do
            let column = sample |> Array.map(fun obs -> obs.Observation.[dim])
            let min, max = Array.min column, Array.max column
            let stepSize = (max - min) / numSteps
            for threshold in min .. stepSize .. max do
                for op in [ (<); (>) ] do
                    let stump = stumpClassify dim threshold op
                    let error =
                        Array.map2 (fun example w -> 
                            weightedError example w stump) sample weights
                        |> Array.sum
                    yield stump, error }
    |> Seq.minBy (fun (stump, err) -> err)

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
        let stump, err = buildStump sample weights
        printfn "Error: %f" err
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

    // Initiate recursive update and create classifier from stumps
    let model = update 0 [] weights
    classify model

//// Example from the book
//let dataset, classLabels = 
//    [| [| 1.0; 2.1 |];
//       [| 2.0; 1.1 |];
//       [| 1.3; 1.0 |];
//       [| 1.0; 1.0 |];
//       [| 2.0; 1.0 |] |], 
//    [| 1.0; 1.0; -1.0; -1.0; 1.0 |]
//
//let classifier = train dataset classLabels 5 0.05
//Array.zip dataset classLabels 
//|> Array.iter (fun (d, l) -> printfn "Real %f Pred %f" l (classifier d))

// Wine classification
// http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

open System.IO
open System.Net

// retrieve data from UC Irvine Machine Learning repository
let url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
let request = WebRequest.Create(url)
let response = request.GetResponse()

let stream = response.GetResponseStream()
let reader = new StreamReader(stream)
let data = reader.ReadToEnd()
reader.Close()
stream.Close()

let parse (line: string) =
    let parsed = line.Split(',') 
    let observation = 
        parsed
        |> Seq.skip 1
        |> Seq.map (float)
        |> Seq.toArray
    let label =
        parsed
        |> Seq.head
        |> (int)
    observation, label

// classifier: 1s vs. rest of the world
let dataset, labels = 
    data.Split((char)10)
    |> Array.filter (fun l -> l.Length > 0) // because of last line
    |> Array.map parse
    |> Array.map (fun (data, l) -> 
        data, if l = 1 then 1.0 else -1.0 )
    |> Array.unzip

let wineClassifier = train dataset labels 10 0.01
Array.zip dataset labels 
|> Array.iter (fun (d, l) -> printfn "Real %f Pred %f" l (wineClassifier d))