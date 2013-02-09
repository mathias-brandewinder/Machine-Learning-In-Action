// Chapter 10: K-means clustering

type Distance<'a> = 'a -> 'a -> float
type CentroidsFactory<'a> = 'a seq -> int -> 'a seq
type ToCentroid<'a> = 'a seq -> 'a

// Returns the index and distance of the 
// centroid closest to observation
let closest (dist: Distance<'a>) centroids (obs: 'a) =
    centroids
    |> Seq.mapi (fun i c -> (i, dist c obs)) 
    |> Seq.minBy (fun (i, d) -> d)

// Euclidean distance between 2 points, represented as float []
let euclidean x y = 
    Array.fold2 (fun d e1 e2 -> d + pown (e1 - e2) 2) 0. x y 
    |> sqrt

// Picks k random observations as initial centroids
let randomCentroids<'a> (rng: System.Random) 
                        (sample: 'a seq) 
                        k =
    let size = Seq.length sample
    seq {
        for i in 1 .. k do 
        let pick = Seq.nth (rng.Next(size)) sample
        yield pick }

let toCentroid (sample: float [] seq) =
    let size = Seq.length sample
    sample
    |> Seq.reduce (fun v1 v2 -> 
           Array.map2 (fun v1x v2x -> v1x + v2x) v1 v2)
    |> Array.map (fun e -> e / (float)size)

let kmeans (dist: Distance<'a>) 
           (factory: CentroidsFactory<'a>) 
           (aggregator: ToCentroid<'a>)
           (dataset: 'a seq) 
           k =
    let initialCentroids = factory dataset k

    let rec update (centroids, assignment) =
        let next = dataset |> Seq.map (fun obs -> closest dist centroids obs)
        let change =
            match assignment with
            | Some(previous) -> 
                Seq.zip previous next    
                |> Seq.exists (fun ((i, _), (j, _)) -> not (i = j))
            | None -> true
        if change 
        then 
            let updated =
                centroids 
                |> Seq.mapi (fun i c -> 
                    Seq.zip dataset next 
                    |> Seq.filter (fun (_, (ci, _)) -> ci = i)
                    |> Seq.map (fun (obs, _) -> obs)
                    |> aggregator)
            (updated, next)
        else (centroids, next)
    update (initialCentroids, None)

// test data
let rng = new System.Random()
let data = [ 
    for i in 1 .. 20 -> [| rng.NextDouble(); rng.NextDouble() |]
    for i in 1 .. 20 -> [| 10. + rng.NextDouble(); 20. + rng.NextDouble() |] ]

let fac = randomCentroids<float[]> rng
let check = kmeans euclidean fac toCentroid data 2