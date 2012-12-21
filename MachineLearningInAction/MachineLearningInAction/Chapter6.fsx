#load "SupportVectorMachine.fs"
#r @"..\..\MachineLearningInAction\packages\MSDN.FSharpChart.dll.0.60\lib\MSDN.FSharpChart.dll"
#r "System.Windows.Forms.DataVisualization"

open MachineLearning.SupportVectorMachine
open System
open System.Drawing
open System.Windows.Forms.DataVisualization
open MSDN.FSharp.Charting
 
let rng = new Random()
let linearKernel = dot

// tight dataset: there is no margin between 2 groups
let tightData = 
    [| for i in 1 .. 500 -> [ rng.NextDouble() * 100.0; rng.NextDouble() * 100.0 ] |]
let tightLabels = 
    tightData |> Array.map (fun el -> 
        if (el |> List.sum >= 100.0) then 1.0 else -1.0)

// loose dataset: there is empty "gap" between 2 groups
let looseData = 
    tightData 
    |> Array.filter (fun e -> 
        let tot = List.sum e
        tot > 110.0 || tot < 90.0)
let looseLabels = 
    looseData |> Array.map (fun el -> 
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
scatterplot (tightData |> Array.map (fun e -> e.[0], e.[1])) tightLabels
scatterplot (looseData |> Array.map (fun e -> e.[0], e.[1])) looseLabels

let test (data: float list []) (labels: float []) parameters =
    let classify = smoClassifier data labels linearKernel parameters
    let performance = 
        data
        |> Array.map (fun row -> classify row)
        |> Array.zip labels
        |> Array.map (fun (a, b) -> if a * b > 0.0 then 1.0 else 0.0)
        |> Array.average
    printfn "Proportion correctly classified: %f" performance

let plot (data: float list []) (labels: float []) parameters =
    let estimator = smo data labels linearKernel parameters
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

let parameters = { C = 0.6; Tolerance = 0.001; Depth = 50 }

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

let plotLine (data: float list []) (labels: float []) parameters =
    let estimator = smo data labels linearKernel parameters
    let w = weights (fst estimator)
    let b = snd estimator
    let line x = - b / w.[1] - x * w.[0] / w.[1]
    separator (data |> Seq.map (fun e -> e.[0], e.[1])) labels line

plotLine tightData tightLabels parameters
plotLine looseData looseLabels parameters

// larger set (1000 observations, 10 dimensions)
let largeData = 
    [| for i in 1 .. 1000 -> [ for d in 1 .. 10 -> rng.NextDouble() * 100.0 ] |]
let largeLabels = 
    largeData |> Array.map (fun x -> 
        if (x |> List.sum >= 500.0) then 1.0 else -1.0)

largeLabels 
    |> Array.filter (fun l -> l = 1.0) 
    |> Array.length 
    |> printfn "Number in group 1: %i"

test largeData largeLabels parameters


// Test case for non-linear data
let circleData = 
    [| for i in 1 .. 500 -> [ rng.NextDouble() * 100.0; rng.NextDouble() * 100.0 ] |]
let dist (coord1: float list) coord2 = 
    List.map2 (fun c1 c2 -> (c1-c2)*(c1-c2)) coord1 coord2
    |> List.sum
    |> sqrt
let center = [ 40.0; 60.0 ]
let circleLabels = 
    circleData 
        |> Array.map (fun data -> dist data center)
        |> Array.map (fun dist -> if dist >= 30.0 then 1.0 else -1.0)

scatterplot (circleData |> Array.map (fun data -> data.[0], data.[1])) circleLabels


// noisy dataset: a percentage of observations is mis-labeled
// Commented out: the classifier seems to really struggle with this.
//let misclassified = 0.05
//let noisyData = tightData
//let noisyLabels = 
//    tightLabels |> Array.map (fun l -> 
//        if (rng.NextDouble() > 1.0 - misclassified) then -l else l)
//
//scatterplot (noisyData |> Array.map (fun e -> e.[0], e.[1])) noisyLabels
//plot noisyData noisyLabels parameters
//test noisyData noisyLabels parameters
//plotLine noisyData noisyLabels parameters

