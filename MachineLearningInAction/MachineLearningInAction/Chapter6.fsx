// Support Vector Machine

open System

let testData =
    [ [ 1.0; 2.0 ];
      [ 3.0; 4.0 ];
      [ 5.0; 6.0 ] ];

let testLabels = [ 1.0; -1.0; 1.0 ]

let clip min max x =
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

let pickAnother (rng: System.Random) i max = 
    rng.Next(0, max)
//    let rec attempt =
//        let j = rng.Next(0, max)
//        if j = i then attempt () else j
//    attempt
         
let simpleSmo dataset (labels: float list) C tolerance iterations =
    
    let size = dataset |> List.length
    
    let b = 0.0
    let alphas = [ for i in 1 .. size -> 0.0 ]

    let rng = new Random()

    let update i = 

            let iClass = labels.[i]
            let iError = (predict dataset labels alphas b i) - iClass
            let iAlpha = alphas.[i]

            if (iError * iClass < - tolerance && iAlpha < C) || (iError * iClass > tolerance && iAlpha > 0.0)
            then
                let j = pickAnother rng i size
                let jClass = labels.[j]
                let jError = (predict dataset labels alphas b j) - jClass
                let jAlpha = alphas.[j]

                let lo, hi = 
                    if labels.[i] = labels.[j]
                    then max 0.0 (iAlpha + jAlpha - C), min C (jAlpha - iAlpha)
                    else max 0.0 (jAlpha - iAlpha),     min C (C + jAlpha - iAlpha) 
                
                if lo = hi 
                then 
                    printfn "Low = High"
                    b, alphas
                else
                    let iObs, jObs = dataset.[i], dataset.[j]
                    let eta = 2.0 * dot iObs jObs - dot iObs iObs - dot jObs jObs

                    if eta >= 0.0 
                    then 
                        printfn "ETA >= 0"
                        b, alphas
                    else
                        let jAlphaNew = clip (jAlpha - (jClass * (iError - jError) / eta)) lo hi

                        if abs (jAlpha - jAlphaNew) < 0.00001 
                        then
                            printfn "j not moving enough"
                            b, alphas
                        else
                            let iAlphaNew = iAlpha + (iClass * jClass * (jAlpha - jAlphaNew))

                            let b1 = b - iError - iClass * (iAlphaNew - iAlpha) * (dot iObs iObs) - jClass * (jAlphaNew - jAlpha) * (dot iObs jObs)
                            let b2 = b - jError - iClass * (iAlphaNew - iAlpha) * (dot iObs jObs) - jClass * (jAlphaNew - jAlpha) * (dot jObs jObs)

                            let bNew =
                                if (iAlphaNew > 0.0 && iAlphaNew < C)
                                then b1
                                elif (jAlphaNew > 0.0 && jAlphaNew < C)
                                then b2
                                else (b1 + b2) / 2.0

                            printfn "Changed %i and %i" i j
                            b, alphas
            else b, alphas

    for i in 0 .. (size - 1) do
        let b, a = update i
        printfn "Updated"