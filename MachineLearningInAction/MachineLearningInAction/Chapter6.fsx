// Support Vector Machine

open System

let rng = new Random()
let testData = [ for i in 1 .. 200 -> [ rng.NextDouble(); rng.NextDouble() ] ]
let testLabels = testData |> List.map (fun el -> if (el |> List.sum >= 0.5) then 1.0 else -1.0)

let clip (min, max) x =
    if (x > max)
    then max
    elif (x < min)
    then min
    else x
    
let dot (vec1: float list) 
        (vec2: float list) =
    List.zip vec1 vec2
    |> List.map (fun e -> fst e * snd e)
    |> List.sum

let elementProduct (alphas: float list) labels =
    List.zip alphas labels |> List.map (fun (a, l) -> a * l)

let predict (data: list<float list>) labels alphas b i =
    let row = data.[i] 
    data 
    |> List.map (fun obs -> dot obs row)
    |> dot (elementProduct alphas labels)
    |> (+) b

// pick an index other than i in [0..(count-1)]
let pickAnother (rng: System.Random) i count = 
    let j = rng.Next(0, count - 1)
    if j >= i then j + 1 else j

let findLowHigh low high (label1, alpha1) (label2, alpha2) = 
    if label1 = label2
    then max low (alpha1 + alpha2 - high), min high (alpha2 - alpha1)
    else max low (alpha2 - alpha1),        min high (high + alpha2 - alpha1) 

type Attempt<'a> = Success of 'a | Failure

let pivot dataset (labels: float list) C tolerance (alphas, b) i j =
    
    printfn "%i %i" i j
    let lohi = findLowHigh 0.0 C

    let iClass = labels.[i]
    let iError = (predict dataset labels alphas b i) - iClass
    let iAlpha = alphas.[i]

    if not (iError * iClass < - tolerance && iAlpha < C) || (iError * iClass > tolerance && iAlpha > 0.0)
    then Failure
    else
        let jClass = labels.[j]
        let jError = (predict dataset labels alphas b j) - jClass
        let jAlpha = alphas.[j]

        let lo, hi = lohi (labels.[i], iAlpha) (labels.[j], jAlpha)

        if lo = hi 
        then Failure
        else
            let iObs, jObs = dataset.[i], dataset.[j]
            let eta = 2.0 * dot iObs jObs - dot iObs iObs - dot jObs jObs
            
            if eta >= 0.0 
            then Failure
            else   
                let jTemp = jAlpha - (jClass * (iError - jError) / eta)
                printfn "%f" jTemp

                let jAlphaNew = clip (lo, hi) (jAlpha - (jClass * (iError - jError) / eta))
                let iAlphaNew = iAlpha + (iClass * jClass * (jAlpha - jAlphaNew))

                printfn "First: %f -> %f" iAlpha iAlphaNew
                printfn "Second: %f -> %f" jAlpha jAlphaNew

                let b1 = b - iError - iClass * (iAlphaNew - iAlpha) * (dot iObs iObs) - jClass * (jAlphaNew - jAlpha) * (dot iObs jObs)
                let b2 = b - jError - iClass * (iAlphaNew - iAlpha) * (dot iObs jObs) - jClass * (jAlphaNew - jAlpha) * (dot jObs jObs)

                let bNew =
                    if (iAlphaNew > 0.0 && iAlphaNew < C)
                    then b1
                    elif (jAlphaNew > 0.0 && jAlphaNew < C)
                    then b2
                    else (b1 + b2) / 2.0
                Success(alphas 
                |> List.mapi (fun index value -> 
                    if index = i 
                    then iAlphaNew 
                    elif index = j then jAlphaNew 
                    else value), bNew)

let nextAround size i = (i + 1) % size

let simpleSvm dataset (labels: float list) C tolerance iterations =
    
    let size = dataset |> List.length   
     
    let b = 0.0
    let alphas = [ for i in 1 .. size -> 0.0 ]

    let rng = new Random()
    let lohi = findLowHigh 0.0 C
    let next i = nextAround size i
    
    let rec search current noChange i =
        if noChange < iterations
        then
            let j = pickAnother rng i size
            let updated = pivot dataset labels C tolerance current i j
            match updated with
            | Failure -> search current (noChange + 1) (next i)
            | Success(result) -> search result 0 (next i)
        else
            current

    search (alphas, b) 0 0

simpleSvm testData testLabels 1.0 0.01 100