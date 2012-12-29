#load "AdaBoost.fs"
open MachineLearning.AdaBoost

open System
open System.IO
open System.Net

// Example from the book
let testDataset, testLabels = 
    [| [| 1.0; 2.1 |];
       [| 2.0; 1.1 |];
       [| 1.3; 1.0 |];
       [| 1.0; 1.0 |];
       [| 2.0; 1.0 |] |], 
    [| 1.0; 1.0; -1.0; -1.0; 1.0 |]

let classifier = train testDataset testLabels 5 10.0 0.05
Array.zip testDataset testLabels 
|> Array.iter (fun (d, l) -> printfn "Real %f Pred %f" l (classifier d))

// Wine classification
// http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data

// Retrieve data from UC Irvine Machine Learning repository
let url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
let request = WebRequest.Create(url)
let response = request.GetResponse()

let stream = response.GetResponseStream()
let reader = new StreamReader(stream)
let data = reader.ReadToEnd()
reader.Close()
stream.Close()

// Prepare data for analysis
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

// We will classify group 1 vs. rest of the world (class 2 and 3)
let dataset, labels = 
    data.Split((char)10)
    |> Array.filter (fun l -> l.Length > 0) // because of last line
    |> Array.map parse
    |> Array.map (fun (data, l) -> 
        data, if l = 1 then 1.0 else -1.0 )
    |> Array.unzip

let group1, group2 =   
    Array.zip dataset labels
    |> Array.partition (fun e -> snd e = 1.0)

let size1 = Array.length group1 / 2
let size2 = Array.length group2 / 2

let trainingSet, trainingLabels = 
    Array.append group1.[ 0 .. size1 ] group2.[ 0 .. size2 ]
    |> Array.unzip

let validation = 
    Array.append group1.[ size1 + 1 .. ] group2.[ size2 + 1 .. ]

let wineClassifier = train trainingSet trainingLabels 20 10.0 0.01

validation 
|> Array.iter (fun (d, l) -> printfn "Real %f Pred %f" l (wineClassifier d))