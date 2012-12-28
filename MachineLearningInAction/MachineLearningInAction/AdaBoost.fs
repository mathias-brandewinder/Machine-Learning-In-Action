namespace MachineLearning

module AdaBoost =

    open System

    type Example = { Observation: float []; Label: float }
    type WeakLearner = { Alpha: float; Classifier: float [] -> float }

    let normalize (weights: float []) = 
        let total = weights |> Array.sum 
        Array.map (fun w -> w / total) weights
