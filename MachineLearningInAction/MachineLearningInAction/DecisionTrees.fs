namespace MachineLearning

open System

type Tree = 
    | Conclusion of string 
    | Choice of string * (string * Tree) []

module DecisionTrees =

    let testData =
        [| "A"; "B"; "C" |],
        [| [| "Yes"; "Yes"; "Yes" |];
           [| "Yes"; "Yes"; "Yes" |];
           [| "Yes"; "No";  "No"  |];
           [| "No";  "Yes"; "No"  |];
           [| "No";  "Yes"; "No"  |] |]

    let prop count total = (float)count / (float)total

    let h vector =
        let size = vector |> Array.length
        vector 
        |> Seq.groupBy (fun e -> e)
        |> Seq.sumBy (fun e ->
            let count = e |> snd |> Seq.length
            let p = prop count size
            - p * Math.Log(p, 2.0))

    let shannonEntropy dataset =
        let size = Array.length dataset
        let cols = dataset.[0] |> Array.length
        dataset
        |> Seq.groupBy (fun row -> row.[cols-1])
        |> Seq.sumBy (fun group -> 
            let p = prop (Seq.length (snd group)) size
            - p * Math.Log(p, 2.0))

    let remove i vector =
        let size = vector |> Array.length
        Array.append vector.[0 .. i-1] vector.[i+1 .. size-1]

    let split dataset i =
        let header, (data: 'a [][]) = dataset
        remove i header,
        data
        |> Seq.groupBy (fun row -> row.[i])
        |> Seq.map (fun group -> 
            group |> snd |> Seq.map (remove i) |> Seq.toArray)

    let splitEntropy dataset i =
        let headers, (data: 'a [][]) = dataset
        let size = data |> Array.length
        let feat = headers |> Array.length
        data
        |> Seq.groupBy(fun row -> row.[i])
        |> Seq.map (fun subset -> 
            subset |> snd |> Seq.map (fun row -> row.[feat - 1]))
        |> Seq.map (fun subset -> subset |> Seq.toArray)
        |> Seq.sumBy (fun subset -> 
            let p = prop (Array.length subset) size
            p * h subset)

    let selectSplit dataset =
        let headers, data = dataset
        let size = data |> Array.length
        let feat = headers |> Array.length
        let currentEntropy = shannonEntropy data
        
        let feature =
            headers.[0 .. feat - 2]
            |> Array.mapi (fun i f ->
                (i, f), currentEntropy - splitEntropy dataset i)
            |> Array.maxBy (fun f -> snd f)
        if (snd feature > 0.0) then Some(fst feature) else None