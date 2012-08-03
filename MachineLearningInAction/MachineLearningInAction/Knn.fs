namespace MachineLearning

open System.Drawing
open MSDN.FSharp.Charting

module Knn =

    let distance v1 v2 =
        Array.zip v1 v2
        |> Array.fold (fun sum e -> sum + pown (fst e - snd e) 2) 0.0
        |> sqrt

    let classify subject dataSet labels k =
        dataSet
        |> Array.map (fun row -> distance row subject)
        |> Array.zip labels
        |> Array.sortBy snd
        |> Array.toSeq
        |> Seq.take k
        |> Seq.groupBy fst
        |> Seq.maxBy (fun g -> Seq.length (snd g))

    let display (dataSet: float[][]) (labels: string []) i j =

        let byLabel =
            dataSet
            |> Array.map (fun e -> e.[i], e.[j])
            |> Array.zip labels

        let uniqueLabels = Seq.distinct labels

        FSharpChart.Combine 
            [ for l in uniqueLabels ->
                    let data = 
                        Array.filter (fun e -> l = fst e) byLabel
                        |> Array.map snd
                    FSharpChart.Point(data) :> ChartTypes.GenericChart
                    |> FSharpChart.WithSeries.DataPoint(Label=l)
            ]
        |> FSharpChart.Create    

    let column (dataset: float [][]) i = 
        dataset |> Array.map (fun row -> row.[i])

    let columns (dataset: float [][]) =
        let cols = dataset.[0] |> Array.length
        [| for i in 0 .. (cols - 1) -> column dataset i |]

    let minMax dataset =
        dataset 
        |> columns 
        |> Array.map (fun col -> Array.min(col), Array.max(col))

    let minMaxNormalizer dataset =
       let bounds = minMax dataset
       fun (vector: float[]) -> 
           Array.mapi (fun i v -> 
               (vector.[i] - fst v) / (snd v - fst v)) bounds

    let normalize data (normalizer: float[] -> float[]) =
        data |> Array.map normalizer

    let classifier dataset labels k =
        let normalizer = minMaxNormalizer dataset
        let normalized = normalize dataset normalizer
        fun subject -> classify (normalizer(subject)) normalized labels k