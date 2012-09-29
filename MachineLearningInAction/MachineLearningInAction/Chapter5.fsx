#load "LogisticRegression.fs"
#r @"C:\Users\Mathias\Documents\GitHub\Machine-Learning-In-Action\MachineLearningInAction\packages\MSDN.FSharpChart.dll.0.60\lib\MSDN.FSharpChart.dll"
#r "System.Windows.Forms.DataVisualization"
open MachineLearning.LogisticRegression

open System.Drawing
open System.Windows.Forms.DataVisualization
open MSDN.FSharp.Charting

#time

// illustration on small example
let testSet =
    [ [ 0.5 ; 0.7 ];
      [ 1.5 ; 2.3 ];
      [ 0.8 ; 0.8 ];
      [ 6.0 ; 9.0 ];
      [ 9.5 ; 5.5 ];     
      [ 6.5 ; 2.7 ];
      [ 2.1 ; 0.1 ];
      [ 3.2 ; 1.9 ] ]
let testLabels = [ 1.0 ; 1.0 ; 1.0; 1.0; 0.0 ; 0.0; 0.0; 0.0 ]
let dataset = Seq.zip testLabels testSet

// compute weights on 1000 iterations, with alpha = 0.1
let estimates = simpleTrain dataset 100 0.1
let classifier = predict estimates

// display dataset, and "separating line"
let display (dataSet: (float * float) seq) (labels: string seq) (line: float -> float) =
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

let xy = testSet |> Seq.map (fun e -> e.[0], e.[1])
let labels = testLabels |> Seq.map (fun e -> e.ToString())
let line x = - estimates.[0] / estimates.[2] - x * estimates.[1] / estimates.[2]
let show = display xy labels line

// comparison of training methods on larger dataset
let rng = new System.Random()
let w0, w1, w2, w3, w4 = 1.0, 2.0, 3.0, 4.0, -10.0
let weights = [ w0; w1; w2; w3; w4 ] // "true" vector
let sampleSize = 10000

let fakeData = 
    [ for i in 1 .. sampleSize -> [ for coord in 1 .. 4 -> rng.NextDouble() * 10.0 ] ]

let inClass x = if x <= 0.5 then 0.0 else 1.0 

let cleanLabels =
    fakeData 
    |> Seq.map (fun coords -> predict weights coords)
    |> Seq.map inClass

let noisyLabels = 
    fakeData 
    |> Seq.map (fun coords -> 
        if rng.NextDouble() < 0.9 
        then predict weights coords
        else rng.NextDouble())
    |> Seq.map inClass

let quality classifier dataset = 
    dataset
    |> Seq.map (fun (lab, coords) -> 
        if lab = (predict classifier coords |> inClass) then 1.0 else 0.0)
    |> Seq.average

printfn "Clean dataset"
let cleanSet = Seq.zip cleanLabels fakeData
printfn "Running simple training"
let clean1 = simpleTrain cleanSet 100 0.1
printfn "Correctly classified: %f" (quality clean1 cleanSet)
printfn "Running convergence-based training"
let clean2 = train cleanSet 0.000001
printfn "Correctly classified: %f" (quality clean2 cleanSet)

printfn "Noisy dataset"
let noisySet = Seq.zip noisyLabels fakeData
printfn "Running simple training"
let noisy1 = simpleTrain noisySet 100 0.1
printfn "Correctly classified: %f" (quality noisy1 noisySet)
printfn "Running convergence-based training"
let noisy2 = train noisySet 0.000001
printfn "Correctly classified: %f" (quality noisy2 noisySet)
