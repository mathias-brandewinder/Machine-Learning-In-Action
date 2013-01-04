namespace MachineLearning

module SupportVectorMachine =

    open System

    // An observation, its label, and current alpha estimate
    type SupportVector = { Data: float list; Label: float; Alpha: float }

    // A Kernel transforms 2 data points into a float
    type Kernel = float list -> float list -> float

    // SVM algorithm input parameters
    type Parameters = { Tolerance: float; C: float; Depth: int }

    // Describes the two algorithm loop types
    type Loop = Full | Subset
    let switch loop = 
        match loop with 
        | Full   -> Subset 
        | Subset -> Full

    // http://en.wikibooks.org/wiki/F_Sharp_Programming/Computation_Expressions
    type MaybeBuilder() =
        member this.Bind(x, f) =
            match x with
            | Some(x) -> f(x)
            | _       -> None
        member this.Delay(f) = f()
        member this.Return(x) = Some x

    // Limit for what is considered too small a change
    let smallChange = 0.00001

    // Product of vectors
    let dot (vec1: float list) 
            (vec2: float list) =
        List.fold2 (fun acc v1 v2 -> 
            acc + v1 * v2) 0.0 vec1 vec2

    // distance between vectors
    let dist (vec1: float list) 
             (vec2: float list) =
        List.fold2 (fun acc v1 v2 -> 
            acc + (v1 - v2) * (v1 - v2)) 0.0 vec1 vec2

    // radial bias function
    let rbf sig2 x = exp ( - x / sig2 )

    // radial bias kernel
    let radialBias sigma 
                   (vec1: float list) 
                   (vec2: float list) =
        rbf (sigma * sigma) (dist vec1 vec2)

    // Clip a value x that is out of the min/max bounds
    let clip (min, max) x =
        if (x > max)
        then max
        elif (x < min)
        then min
        else x

    // Identify whether a support vector is "bound"
    let isBound parameters sv = sv.Alpha <= 0.0 || sv.Alpha >= parameters.C

    // Identify bounds of acceptable Alpha changes
    let findLowHigh (low, high) sv1 sv2 = 
        if sv1.Label = sv2.Label
        then max low (sv1.Alpha + sv2.Alpha - high), min high (sv2.Alpha + sv1.Alpha)
        else max low (sv2.Alpha - sv1.Alpha),        min high (high + sv2.Alpha - sv1.Alpha) 
    
    // Compute limits of Alpha change, if possible
    let changeBounds (low, high) sv1 sv2 =
        let l, h = findLowHigh (low, high) sv1 sv2
        if h > l then Some(l, h) else None

    // Compute error on support vector, given current alphas
    let error (rows, b) kernel sv =
        rows
        |> Seq.filter (fun r -> r.Alpha > 0.0)
        |> Seq.fold (fun acc r -> 
            acc + r.Label * r.Alpha * (kernel r.Data sv.Data)) (b - sv.Label)

    // Check whether alpha can be modified for support vector,
    // given current prediction error and parameter C
    let canChange parameters sv error =
        (error * sv.Label < - parameters.Tolerance && sv.Alpha < parameters.C)
        || (error * sv.Label > parameters.Tolerance && sv.Alpha > 0.0)

    // Utility function, not sure how to call that guy
    let f kernel sv1 sv2 alpha1 alpha2 =
        sv1.Label * (alpha1 - sv1.Alpha) * (kernel sv1.Data sv2.Data) + 
        sv2.Label * (alpha2 - sv2.Alpha) * (kernel sv1.Data sv2.Data)

    // Update the model constant b
    let updateB kernel b sv1 sv2 alpha1' alpha2' iError jError C =
        let b1 = b - iError - f kernel sv1 sv2 alpha1' alpha2' 
        let b2 = b - jError - f kernel sv2 sv1 alpha2' alpha1' 

        if (0.0 < alpha1' && alpha1' < C)
        then b1
        elif (0.0 < alpha2' && alpha2' < C)
        then b2
        else (b1 + b2) / 2.0
     
    let computeEta kernel sv1 sv2 =
        let eta = 2.0 * kernel sv1.Data sv2.Data - kernel sv1.Data sv1.Data - kernel sv2.Data sv2.Data
        if eta >= 0.0 then None else Some(eta)

    // Pick a random index other than i in  [0 .. (count-1) ]
    let pickAnother (rng: System.Random) i count = 
        let j = rng.Next(0, count - 1)
        if j >= i then j + 1 else j
    
    // Find a suitable second vector to pivot with support vector i 
    let identifyCandidate (svs: SupportVector []) b kernel (rng: Random) parameters i =

        let sv1 = svs.[i]
        let error1 = error (svs, b) kernel sv1

        match (canChange parameters sv1 error1) with
        | false -> None
        | true  ->
            let candidates =
                Seq.mapi (fun i sv -> (i, sv)) svs
                |> Seq.filter (fun (i, sv) -> not (isBound parameters sv))
                |> Seq.toArray
            if (Array.length candidates > 1)
            then // Return support vector with largest error difference
                let (j, sv2, error2) =
                    candidates
                    |> Seq.map (fun (i, sv) -> (i, sv, error (svs, b) kernel sv))
                    |> Seq.maxBy (fun (i, sv, e) -> abs (error1 - e))
                Some((i, sv1, error1), (j, sv2, error2))
            else // Try some random guy
                let j = pickAnother rng i (Array.length svs)
                let sv2 = svs.[j]
                let error2 = error (svs, b) kernel sv2
                Some((i, sv1, error1), (j, sv2, error2))

    let maybe = MaybeBuilder()

    // Attempt to pivot a pair of support vectors (current errors pre-computed)
    let pivotPair (svs: SupportVector []) b kernel parameters data1 data2 =
        let (i, sv1, error1) = data1
        let (j, sv2, error2) = data2
        maybe { 
            let! lo, hi = changeBounds (0.0, parameters.C) sv1 sv2
            let! eta = computeEta kernel sv1 sv2            
            let! alpha2' = 
                let candidate = clip (lo, hi) (sv2.Alpha - (sv2.Label * (error1 - error2) / eta))
                if (abs (candidate - sv2.Alpha) < smallChange) then None else Some(candidate)
            // the pivot is succesfull: proceed with updated alphas
            let alpha1' = sv1.Alpha + (sv1.Label * sv2.Label * (sv2.Alpha - alpha2'))
            let b' = updateB kernel b sv1 sv2 alpha1' alpha2' error1 error2 parameters.C

            let updatedSvs =
                svs 
                |> Array.mapi (fun index value -> 
                    if index = i 
                    then { value with Alpha = alpha1' } 
                    elif index = j 
                    then { value with Alpha = alpha2' }
                    else value)
            return (updatedSvs, b') }

    // Sequential Minimal Optimization
    let smo dataset labels (kernel: Kernel) parameters = 
        // Data preparation
        let size = dataset |> Array.length        
        let initialSVs = 
            Array.zip dataset labels
            |> Array.map (fun (d, l) -> { Data = d; Label = l; Alpha = 0.0 })
        let b = 0.0
        let rng = new Random()

        // Search: recursively loop over observations, updating alphas
        let rec search (current: SupportVector []) b iter loop =
            let pivots =
                match loop with
                | Full   -> seq { 0 .. size - 1 }
                | Subset -> 
                    seq { 0 .. size - 1 } 
                    |> Seq.filter (fun i -> not (isBound parameters current.[i]))
            let changes = 0 // count pairs changed in loop
            // Update support vectors and b, by attempting a pivot on every index
            let updated =
                Seq.fold (fun (svs, b, chs) index -> 
                    match (identifyCandidate svs b kernel rng parameters index) with
                    | None           -> (svs, b, chs)
                    | Some(sv1, sv2) -> 
                        match (pivotPair svs b kernel parameters sv1 sv2) with
                        | None           -> (svs, b, chs)
                        | Some(svs', b') -> (svs', b', chs + 1))
                        (current, b, changes) pivots
            let (svs, b, changes) = updated

            if (changes = 0 && loop = Subset) 
            then 
                printfn "Convergence reached: no changes in Supports."
                (svs, b)
            elif (iter > parameters.Depth) 
            then 
                printfn "No convergence, iteration limit reached."
                (svs, b)
            else search svs b (iter + 1) (switch loop)

        // run search
        search initialSVs b 0 Full

    // Compute the weights, using support vectors returned from SVM
    // (Useful to simplify final model with linear Kernel)
    let weights svs =
        svs 
        |> Seq.filter (fun sv -> sv.Alpha > 0.0)
        |> Seq.map (fun sv ->
            let mult = sv.Alpha * sv.Label
            sv.Data |> List.map (fun e -> mult * e))
        |> Seq.reduce (fun acc row -> 
            List.map2 (fun a r -> a + r) acc row )
    
    // Create a classifier function from SVM results
    let classifier (kernel: Kernel) estimator =
        // Classifier function
        fun obs ->         
            let (svs, b) = estimator
            svs
            |> Seq.filter (fun sv -> sv.Alpha > 0.0)
            |> Seq.fold (fun acc sv -> 
                acc + sv.Label * sv.Alpha * (kernel sv.Data obs)) b


