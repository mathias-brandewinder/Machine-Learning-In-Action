#load "SupportVectorMachine.fs"
#r @"C:\Users\Mathias\Documents\GitHub\Machine-Learning-In-Action\MachineLearningInAction\packages\MSDN.FSharpChart.dll.0.60\lib\MSDN.FSharpChart.dll"
#r "System.Windows.Forms.DataVisualization"

open MachineLearning.SupportVectorMachine
open System
open System.Drawing
open System.Windows.Forms.DataVisualization
open MSDN.FSharp.Charting
 
// pick an index other than i in [0..(count-1)]
let pickAnother (rng: System.Random) i count = 
    let j = rng.Next(0, count - 1)
    if j >= i then j + 1 else j

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
    
    let rowi = rows.[i]
    let iError = rowError rows b rowi

    if not (iError * rowi.Label < - parameters.Tolerance && rowi.Alpha < parameters.C) || (iError * rowi.Label > parameters.Tolerance && rowi.Alpha > 0.0)
    then Failure
    else
        let rowj = rows.[j]
        let lo, hi = findLowHigh 0.0 parameters.C rowi rowj

        if lo = hi 
        then Failure
        else
            let eta = 2.0 * dot rowi.Data rowj.Data - dot rowi.Data rowi.Data - dot rowj.Data rowj.Data
            
            if eta >= 0.0 
            then Failure
            else   
                let jError = rowError rows b rowj

                let jAlphaNew = clip (lo, hi) (rowj.Alpha - (rowj.Label * (iError - jError) / eta))
                let iAlphaNew = rowi.Alpha + (rowi.Label * rowj.Label * (rowj.Alpha - jAlphaNew))
                let bNew = updateB b rowi rowj iAlphaNew jAlphaNew iError jError parameters.C

                printfn "First: %f -> %f" rowi.Alpha iAlphaNew
                printfn "Second: %f -> %f" rowj.Alpha jAlphaNew
                printfn "B: %f -> %f" b bNew

                Success(rows 
                |> List.mapi (fun index value -> 
                    if index = i 
                    then { Data = value.Data; Label = value.Label; Alpha = iAlphaNew } 
                    elif index = j 
                    then { Data = value.Data; Label = value.Label; Alpha = jAlphaNew }
                    else value), bNew)

let simpleSvm dataset (labels: float list) C tolerance iterations =
    
    let parameters = { Tolerance = tolerance; C = C }
    let size = dataset |> List.length        
    let b = 0.0

    let rows = 
        List.zip dataset labels
        |> List.map (fun (d, l) -> { Data = d; Label = l; Alpha = 0.0 })

    let rng = new Random()
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
    |> List.map (fun r ->
        let mult = r.Alpha * r.Label
        r.Data |> List.map (fun e -> mult * e))
    |> List.reduce (fun acc row -> 
        List.map2 (fun a r -> a + r) acc row )
        
// test
let rng = new Random()
let testData = [ for i in 1 .. 1000 -> [ rng.NextDouble() * 100.0; rng.NextDouble() * 100.0 ] ]
let testLabels = testData |> List.map (fun el -> if (el |> List.sum >= 100.0) then 1.0 else -1.0)

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

let display (dataSet: (float * float) seq) (labels: int seq) =
    let byLabel = Seq.zip labels dataSet |> Seq.toArray
    let uniqueLabels = Seq.distinct labels
    FSharpChart.Combine 
        [ // separate points by class and scatterplot them
          for label in uniqueLabels ->
               let data = 
                    Array.filter (fun e -> label = fst e) byLabel
                    |> Array.map snd
               FSharpChart.Point(data) :> ChartTypes.GenericChart
               |> FSharpChart.WithSeries.Marker(Size=10)
        ]
    |> FSharpChart.Create    

let testPlot =
    let estimator = simpleSvm testData testLabels 5.0 0.001 500
    let labels = 
        estimator 
        |> (fst) 
        |> Seq.map (fun row -> 
            if row.Alpha > 0.0 then 0
            else
                if row.Label < 0.0 then 1
                else 2)
    let data = 
        estimator 
        |> (fst) 
        |> Seq.map (fun row -> (row.Data.[0], row.Data.[1]))
    display data labels