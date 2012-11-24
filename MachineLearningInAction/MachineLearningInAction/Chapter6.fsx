#load "SupportVectorMachine.fs"
#r @"..\..\MachineLearningInAction\packages\MSDN.FSharpChart.dll.0.60\lib\MSDN.FSharpChart.dll"
#r "System.Windows.Forms.DataVisualization"

open MachineLearning.SupportVectorMachine
open System
open System.Drawing
open System.Windows.Forms.DataVisualization
open MSDN.FSharp.Charting
 
let rng = new Random()

// tight dataset: there is no margin between 2 groups
let tightData = 
    [ for i in 1 .. 500 -> [ rng.NextDouble() * 100.0; rng.NextDouble() * 100.0 ] ]
let tightLabels = 
    tightData |> List.map (fun el -> 
        if (el |> List.sum >= 100.0) then 1.0 else -1.0)

// loose dataset: there is empty "gap" between 2 groups
let looseData = 
    tightData 
    |> List.filter (fun e -> 
        let tot = List.sum e
        tot > 110.0 || tot < 90.0)
let looseLabels = 
    looseData |> List.map (fun el -> 
        if (el |> List.sum >= 100.0) then 1.0 else -1.0)

// create an X,Y scatterplot, with different formatting for each label 
let scatterplot (dataSet: (float * float) seq) (labels: 'a seq) =
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

// plot raw datasets
scatterplot (tightData |> List.map (fun e -> e.[0], e.[1])) tightLabels
scatterplot (looseData |> List.map (fun e -> e.[0], e.[1])) looseLabels

let test (data: float list list) (labels: float list) parameters =
    let classify = classifier data labels parameters
    let performance = 
        data
        |> List.map (fun row -> classify row)
        |> List.zip labels
        |> List.map (fun (a, b) -> if a * b > 0.0 then 1.0 else 0.0)
        |> List.average
    printfn "Proportion correctly classified: %f" performance

let plot (data: float list list) (labels: float list) parameters =
    let estimator = simpleSvm data labels parameters
    let labels = 
        estimator 
        |> (fst) 
        |> Seq.map (fun row -> 
            if row.Alpha > 0.0 then 0
            elif row.Label < 0.0 then 1
            else 2)
    let data = 
        estimator 
        |> (fst) 
        |> Seq.map (fun row -> (row.Data.[0], row.Data.[1]))
    scatterplot data labels

let parameters = { C = 1.0; Tolerance = 0.01; Depth = 500 }

test tightData tightLabels parameters
test looseData looseLabels parameters

plot tightData tightLabels parameters
plot looseData looseLabels parameters

// display dataset, and "separating line"
let separator (dataSet: (float * float) seq) (labels: 'a seq) (line: float -> float) =
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
          // plot line between left- and right-most points
          let x = Seq.map fst dataSet
          let xMin, xMax = Seq.min x, Seq.max x           
          let lineData = [ (xMin, line xMin); (xMax, line xMax)]
          yield FSharpChart.Line (lineData)  :> ChartTypes.GenericChart
        ]
    |> FSharpChart.Create 

let plotLine (data: float list list) (labels: float list) parameters =
    let estimator = simpleSvm data labels parameters
    let w = weights (fst estimator)
    let b = snd estimator
    let line x = - b / w.[1] - x * w.[0] / w.[1]
    separator (data |> Seq.map (fun e -> e.[0], e.[1])) labels line

plotLine tightData tightLabels parameters
plotLine looseData looseLabels parameters

// noisy dataset: a percentage of observations is mis-labeled
let misclassified = 0.01
let noisyData = tightData
let noisyLabels = 
    tightLabels |> List.map (fun l -> 
        if (rng.NextDouble() > 1.0 - misclassified) then -l else l)

let ezParameters = { C = 1.0; Tolerance = 0.1; Depth = 50 }

scatterplot (noisyData |> List.map (fun e -> e.[0], e.[1])) noisyLabels
plot noisyData noisyLabels ezParameters
test noisyData noisyLabels ezParameters
plotLine noisyData noisyLabels ezParameters

// larger set (1000 observations, 10 dimensions)
let largeData = 
    [ for i in 1 .. 1000 -> [ for d in 1 .. 10 -> rng.NextDouble() * 100.0 ] ]
let largeLabels = 
    largeData |> List.map (fun x -> 
        if (x |> List.sum >= 500.0) then 1.0 else -1.0)

largeLabels 
    |> List.filter (fun l -> l = 1.0) 
    |> List.length 
    |> printfn "Number in group 1: %i"

test largeData largeLabels parameters