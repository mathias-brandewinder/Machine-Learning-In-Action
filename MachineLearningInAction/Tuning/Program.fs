open System
open System.IO
open System.Net
open MachineLearning.SupportVectorMachine

let main =

    // retrieve data from UC Irvine Machine Learning repository
    let url = "http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data"
    let request = WebRequest.Create(url)
    let response = request.GetResponse()

    use stream = response.GetResponseStream()
    use reader = new StreamReader(stream)
    let data = reader.ReadToEnd()

    // prepare data

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

    // split dataset into training vs. valiation
    let sampleSize = 600
    let trainingSet = dataset.[ 0 .. (sampleSize - 1)]
    let trainingLbl = labels.[ 0 .. (sampleSize - 1)]
    let validateSet = dataset.[ sampleSize .. ]
    let validateLbl = labels.[ sampleSize .. ]

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

    // calibration (Careful,takes a while)
    //for c in [ 0.1; 1.0; 10.0 ] do
    for c in [ 10.0 ] do
    //    for s in [ 0.1; 1.0; 10.0 ] do
        for s in [ 1.0; 10.0 ] do
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

    Console.WriteLine("[Enter] to close")
    Console.ReadLine()