#load "SupportVectorMachine.fs"

open System.IO
open System.Net
open MachineLearning.SupportVectorMachine

// retrieve data from UC Irvine Machine Learning repository
let url = "http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data"
let request = WebRequest.Create(url)
let response = request.GetResponse()

let stream = response.GetResponseStream()
let reader = new StreamReader(stream)
let data = reader.ReadToEnd()
reader.Close()
stream.Close()

// prepare data

// a line in the dataset is 16 x 16 = 256 pixels,
// followed by 10 digits, 1 denoting the number
let parse (line: string) =
    let parsed = line.Split(' ') 
    let observation = 
        parsed
        |> Seq.take 256
        |> Seq.map (fun s -> (float)s)
        |> Seq.toArray
    let label =
        parsed
        |> Seq.skip 256
        |> Seq.findIndex (fun e -> (int)e = 1)
    observation, label

// renders a scanned digit as "ASCII-art"
let render observation =
    printfn " "
    List.iteri (fun i pix ->
        if i % 16 = 0 then printfn "" |> ignore
        if pix > 0.0 then printf "■" else printf " ") observation

// classifier: 7s vs. rest of the world

let dataset, labels = 
    data.Split((char)10)
    |> Array.filter (fun l -> l.Length > 0) // because of last line
    |> Array.map parse
    |> Array.map (fun (data, l) -> 
        data |> Array.toList, if l = 7 then 1.0 else -1.0 )
    |> Array.unzip

let parameters = { C = 5.0; Tolerance = 0.001; Depth = 20 }
let rbfKernel = radialBias 10.0

// split dataset into training vs. valiation
let sampleSize = 600
let trainingSet = dataset.[ 0 .. (sampleSize - 1)]
let trainingLbl = labels.[ 0 .. (sampleSize - 1)]
let validateSet = dataset.[ sampleSize .. ]
let validateLbl = labels.[ sampleSize .. ]

printfn "Training classifier"
let model = smo trainingSet trainingLbl rbfKernel parameters
let classify = classifier rbfKernel model

// Compute average correctly classified
let quality classifier sample =
    sample
    |> Array.map (fun (d, l) -> if (classifier d) * l > 0.0 then 1.0 else 0.0)
    |> Array.average
    |> printfn "Proportion correctly classified: %f"

// split dataset by label and compute quality for each group
let evaluate classifier (dataset, labels) =
    let group1, group2 =
        Array.zip dataset labels
        |> Array.partition (fun (d, l) -> l > 0.0)
    quality classifier group1
    quality classifier group2

// verify training sample classification
printfn "Classification in training set"
evaluate classify (trainingSet, trainingLbl)

// validate on remaining sample
printfn "Classification in validation set"
evaluate classify (validateSet, validateLbl)

// calibration (Careful,takes a while)
for c in [ 0.1; 1.0; 10.0 ] do
    for s in [ 0.1; 1.0; 10.0 ] do
        let parameters = { C = c; Tolerance = 0.001; Depth = 10 }
        let rbfKernel = radialBias s

        printfn "Model with C = %f, s = %f" c s
        let model = smo trainingSet trainingLbl rbfKernel parameters
        let classify = classifier rbfKernel model

        printfn "Classification in training set"
        evaluate classify (trainingSet, trainingLbl)

        // validate on remaining sample
        printfn "Classification in validation set"
        evaluate classify (validateSet, validateLbl)
        
        printfn "Done"

        
// display the 200 first observations and their classification
Array.sub (Array.zip dataset labels) 0 199
|> Array.iter (fun (d, l) -> 
    printfn ""    
    printfn "******************************************"
    let predicted = if (classify d) > 0.0 then 1.0 else -1.0
    printfn "Class %f, classified as %f" l predicted
    render d)