#load "SupportVectorMachine.fs"

open MachineLearning.SupportVectorMachine
open System
 
let rowError rows b row =
    rows
    |> Seq.filter (fun r -> r.Alpha > 0.0)
    |> Seq.fold (fun acc r -> 
        acc + r.Label * r.Alpha * (dot r.Data row.Data)) (b - row.Label)

// pick an index other than i in [0..(count-1)]
let pickAnother (rng: System.Random) i count = 
    let j = rng.Next(0, count - 1)
    if j >= i then j + 1 else j

let findLowHigh low high row1 row2 = 
    if row1.Label = row2.Label
    then max low (row1.Alpha + row2.Alpha - high), min high (row2.Alpha - row1.Alpha)
    else max low (row2.Alpha - row1.Alpha),        min high (high + row2.Alpha - row1.Alpha) 

let updateB b rowI rowJ iAlphaNew jAlphaNew iError jError C =

    let b1 = b - iError - rowI.Label * (iAlphaNew - rowI.Alpha) * (dot rowI.Data rowI.Data) - rowJ.Label * (jAlphaNew - rowJ.Alpha) * (dot rowI.Data rowJ.Data)
    let b2 = b - jError - rowI.Label * (iAlphaNew - rowI.Alpha) * (dot rowI.Data rowJ.Data) - rowJ.Label * (jAlphaNew - rowJ.Alpha) * (dot rowJ.Data rowJ.Data)

    if (iAlphaNew > 0.0 && iAlphaNew < C)
    then b1
    elif (jAlphaNew > 0.0 && jAlphaNew < C)
    then b2
    else (b1 + b2) / 2.0

let pivot (rows: Row list) b parameters i j =
    
    printfn "%i %i" i j
    let lohi = findLowHigh 0.0 parameters.C
    
    let rowi = rows.[i]
    let iClass = rowi.Label
    let iError = rowError rows b rowi
    let iAlpha = rowi.Alpha

    if not (iError * iClass < - parameters.Tolerance && iAlpha < parameters.C) || (iError * iClass > parameters.Tolerance && iAlpha > 0.0)
    then Failure
    else
        let rowj = rows.[j]
        let jClass = rowj.Label
        let jError = rowError rows b rowj
        let jAlpha = rowj.Alpha

        let lo, hi = lohi rowi rowj

        if lo = hi 
        then Failure
        else
            let iObs, jObs = rowi.Data, rowj.Data
            let eta = 2.0 * dot iObs jObs - dot iObs iObs - dot jObs jObs
            
            if eta >= 0.0 
            then Failure
            else   
                //let jTemp = jAlpha - (jClass * (iError - jError) / eta)
                //printfn "%f" jTemp

                let jAlphaNew = clip (lo, hi) (jAlpha - (jClass * (iError - jError) / eta))
                let iAlphaNew = iAlpha + (iClass * jClass * (jAlpha - jAlphaNew))
                let bNew = updateB b rowi rowj iAlphaNew jAlphaNew iError jError parameters.C

                printfn "First: %f -> %f" iAlpha iAlphaNew
                printfn "Second: %f -> %f" jAlpha jAlphaNew
                printfn "B: %f -> %f" b bNew

                Success(rows 
                |> List.mapi (fun index value -> 
                    if index = i 
                    then { Data = value.Data; Label = value.Label; Alpha = iAlphaNew } 
                    elif index = j 
                    then { Data = value.Data; Label = value.Label; Alpha = jAlphaNew }
                    else value), bNew)

let nextAround size i = (i + 1) % size

let simpleSvm dataset (labels: float list) C tolerance iterations =
    
    let parameters = { Tolerance = tolerance; C = C }

    let size = dataset |> List.length   
     
    let b = 0.0

    let rows = 
        List.zip dataset labels
        |> List.map (fun (d, l) -> { Data = d; Label = l; Alpha = 0.0 })

    let rng = new Random()
    let lohi = findLowHigh 0.0 C
    let next i = nextAround size i
    
    let rec search current noChange i =
        if noChange < iterations
        then
            let j = pickAnother rng i size
            let updated = pivot (fst current) (snd current) parameters i j
            match updated with
            | Failure -> search current (noChange + 1) (next i)
            | Success(result) -> search result 0 (next i)
        else
            current

    search (rows, b) 0 0

let weights rows =
    rows 
    |> List.map (fun r -> r.Alpha, r.Data, r.Label)
    |> List.map (fun (a, row, l) ->
        let mult = a * l
        row |> List.map (fun e -> mult * e))
    |> List.reduce (fun acc row -> 
        List.map2 (fun a r -> a + r) acc row )
        
// test

let rng = new Random()
let testData = [ for i in 1 .. 100 -> [ rng.NextDouble(); rng.NextDouble() ] ]
let testLabels = testData |> List.map (fun el -> if (el |> List.sum >= 0.5) then 1.0 else -1.0)

let test () =
    let estimator = simpleSvm testData testLabels 1.0 0.001 100
    let w = weights (fst estimator)
    let b = snd estimator

    let classify row = b + dot w row
    let performance = 
        testData 
        |> List.map (fun row -> classify row)
        |> List.zip testLabels
        |> List.map (fun (a, b) -> if a * b > 0.0 then 1.0 else 0.0)
        |> List.average
    performance