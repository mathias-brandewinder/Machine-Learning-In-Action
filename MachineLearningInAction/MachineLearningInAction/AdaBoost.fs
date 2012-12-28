namespace MachineLearning

module AdaBoost =

    open System

    // A "known example": an observation and its known class label
    type Example = { Observation: float []; Label: float }
    // A "weak learner": a rudimentary classifier and its weight Alpha
    type WeakLearner = { Alpha: float; Classifier: float [] -> float }

    let normalize (weights: float []) = 
        let total = weights |> Array.sum 
        Array.map (fun w -> w / total) weights
