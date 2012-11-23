namespace MachineLearning

module SupportVectorMachine =

    // an observation, its label, and current alpha estimate
    type Row = { Data: float list; Label: float; Alpha: float }
    // SVM algorithm input parameters
    type Parameters = { Tolerance: float; C: float; Depth: int }

    type Attempt<'a> = Success of 'a | Failure

    // Product of vectors
    let dot (vec1: float list) 
            (vec2: float list) =
        List.fold2 (fun acc v1 v2 -> 
            acc + v1 * v2) 0.0 vec1 vec2

    // Clip a value x that is out of the min/max bounds
    let clip (min, max) x =
        if (x > max)
        then max
        elif (x < min)
        then min
        else x

    let findLowHigh low high row1 row2 = 
        if row1.Label = row2.Label
        then max low (row1.Alpha + row2.Alpha - high), min high (row2.Alpha + row1.Alpha)
        else max low (row2.Alpha - row1.Alpha),        min high (high + row2.Alpha - row1.Alpha) 
    
    // next index "around the clock"
    let nextAround size i = (i + 1) % size

    // compute error on row, given current alphas
    let rowError rows b row =
        rows
        |> Seq.filter (fun r -> r.Alpha > 0.0)
        |> Seq.fold (fun acc r -> 
            acc + r.Label * r.Alpha * (dot r.Data row.Data)) (b - row.Label)

    // check whether alpha can be modified for row,
    // given error and parameter C
    let canChange parameters row error =
        if (error * row.Label < - parameters.Tolerance && row.Alpha < parameters.C)
        then true
        elif (error * row.Label > parameters.Tolerance && row.Alpha > 0.0)
        then true
        else false