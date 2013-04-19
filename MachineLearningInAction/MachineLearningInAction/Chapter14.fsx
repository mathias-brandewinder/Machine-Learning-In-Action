// Chapter 14 covers SVD (Singular Value Decomposition)

#r @"..\..\MachineLearningInAction\packages\MathNet.Numerics.2.4.0\lib\net40\MathNet.Numerics.dll"
#r @"..\..\MachineLearningInAction\packages\MathNet.Numerics.FSharp.2.4.0\lib\net40\MathNet.Numerics.FSharp.dll"

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Statistics

type Rating = { UserId: int; DishId: int; Rating: int }

// Our existing "ratings database"
let ratings = [
    { UserId = 0; DishId = 0; Rating = 2 };
    { UserId = 0; DishId = 3; Rating = 4 };
    { UserId = 0; DishId = 4; Rating = 4 };
    { UserId = 1; DishId = 10; Rating = 5 };
    { UserId = 2; DishId = 7; Rating = 1 };
    { UserId = 2; DishId = 0; Rating = 4 };
    { UserId = 3; DishId = 0; Rating = 3 };
    { UserId = 3; DishId = 1; Rating = 3 };
    { UserId = 3; DishId = 2; Rating = 4 };
    { UserId = 3; DishId = 4; Rating = 3 };
    { UserId = 3; DishId = 7; Rating = 2 };
    { UserId = 3; DishId = 8; Rating = 2 };
    { UserId = 4; DishId = 0; Rating = 5 };
    { UserId = 4; DishId = 1; Rating = 5 };
    { UserId = 4; DishId = 2; Rating = 5 };
    { UserId = 5; DishId = 6; Rating = 5 };
    { UserId = 5; DishId = 9; Rating = 5 };
    { UserId = 6; DishId = 0; Rating = 4 };
    { UserId = 6; DishId = 2; Rating = 4 };
    { UserId = 6; DishId = 10; Rating = 5 };
    { UserId = 7; DishId = 5; Rating = 4 };
    { UserId = 7; DishId = 10; Rating = 4 };
    { UserId = 8; DishId = 6; Rating = 5 };
    { UserId = 9; DishId = 3; Rating = 3 };
    { UserId = 9; DishId = 8; Rating = 4 };
    { UserId = 9; DishId = 9; Rating = 5 };
    { UserId = 10; DishId = 0; Rating = 1 };
    { UserId = 10; DishId = 1; Rating = 1 };
    { UserId = 10; DishId = 2; Rating = 2 };
    { UserId = 10; DishId = 3; Rating = 1 };
    { UserId = 10; DishId = 4; Rating = 1 };
    { UserId = 10; DishId = 5; Rating = 2 };
    { UserId = 10; DishId = 6; Rating = 1 };
    { UserId = 10; DishId = 8; Rating = 4 };
    { UserId = 10; DishId = 9; Rating = 5 } ]

// Let's populate a matrix with these ratings;
// unrated items are denoted by a 0

let rows = 11
let cols = 11
let data = DenseMatrix(rows, cols)
ratings 
|> List.iter (fun rating -> 
       data.[rating.UserId, rating.DishId] <- (float)rating.Rating)

// "Pretty" rendering of a matrix
// something is not 100% right here, but it's good enough
let printNumber v = 
    if v < 0. 
    then printf "%.2f " v 
    else printf " %.2f " v
// Display a Matrix in a "pretty" format
let pretty matrix = 
    Matrix.iteri (fun row col value ->
        if col = 0 then printfn "" else ignore ()
        printNumber value) matrix
    printfn ""

printfn "Original data matrix"
pretty data

// Now let's run a SVD on that matrix
let svd = data.Svd(true)
let U, sigmas, Vt = svd.U(), svd.S(), svd.VT()

// recompose the S matrix from the singular values
printfn "S-matrix, with singular values in diagonal"
let S = DiagonalMatrix(rows, cols, sigmas.ToArray())
pretty S

// The SVD decomposition verifies
// data = U x S x Vt

printfn "Reconstructed matrix from SVD decomposition"
let reconstructed = U * S * Vt
pretty reconstructed

// Can we interpret the SVD decomposition? Let's see.

// Each row maps to a User, 
// each column to an extracted Category
let userToCategory = U * S |> pretty

// Each row maps to an extracted Category, 
// each column to a Dish
let categoryToDish = S * Vt |> pretty

// We can use SVD as a data compression mechanism
// by retaining only the n first singular values

// Total energy is the sum of the squares 
// of the singular values
let totalEnergy = sigmas.DotProduct(sigmas)
// Let's compute how much energy is contributed
// by each singular value
printfn "Energy contribution by Singular Value"
sigmas.ToArray() 
|> Array.fold (fun acc x ->
       let energy = x * x
       let percent = (acc + energy)/totalEnergy
       printfn "Energy: %.1f, Percent of total: %.3f" energy percent
       acc + energy) 0. 
|> ignore

// We'll retain only the first 5 singular values,
// which cover 90% of the total energy
let subset = 5
let U' = U.SubMatrix(0, U.RowCount, 0, subset)
let S' = S.SubMatrix(0, subset, 0, subset)
let Vt' = Vt.SubMatrix(0, subset, 0, Vt.ColumnCount)

// Using U', S', Vt' instead of U, S, Vt
// should produce a "decent" approximation
// of the original matrix

printfn "Approximation of the original matrix"
U' * S' * Vt' |> pretty

// To make recommendations we need a similarity measure
type similarity = Generic.Vector<float> -> Generic.Vector<float> -> float
let euclideanSimilarity (v1: Generic.Vector<float>) v2 =
    1. / (1. + (v1 - v2).Norm(2.))
let cosineSimilarity (v1: Generic.Vector<float>) v2 =
    v1.DotProduct(v2) / (v1.Norm(2.) * v2.Norm(2.))
let pearsonSimilarity (v1: Generic.Vector<float>) v2 =
    if v1.Count > 2 
    then 0.5 + 0.5 * Correlation.Pearson(v1, v2)
    else 1.

// Let's check if that works
// Retrieve the first 2 dishes from the data matrix
let dish0 = data.Column(0)
let dish1 = data.Column(1)

printfn "Euclidean similarity, Dish 1 and Dish 1: %f" (euclideanSimilarity dish0 dish0)
printfn "Euclidean similarity, Dish 1 and Dish 2: %f" (euclideanSimilarity dish0 dish1)
printfn "Cosine similarity, Dish 1 and Dish 1: %f" (cosineSimilarity dish0 dish0)
printfn "Cosine similarity, Dish 1 and Dish 2: %f" (cosineSimilarity dish0 dish1)
printfn "Pearson similarity, Dish 1 and Dish 1: %f" (pearsonSimilarity dish0 dish0)
printfn "Pearson similarity, Dish 1 and Dish 2: %f" (pearsonSimilarity dish0 dish1)

// Naive recommendation

// reduce 2 vectors to their non-zero pairs
let nonZeroes (v1:Generic.Vector<float>) (v2:Generic.Vector<float>) =
    let size = v1.Count
    let overlap =
        [ 0 .. (size - 1) ] 
        |> List.fold (fun acc i -> 
            if v1.[i] > 0. && v2.[i] > 0. 
            then (v1.[i], v2.[i])::acc else acc) []
    match overlap with
    | [] -> None
    | x  -> 
        let v1', v2' =
            x
            |> List.toArray
            |> Array.unzip
        Some(DenseVector(v1'), DenseVector(v2'))

// Retrieve rating of a dish and similarity with another dish
let weightedRating (unrated:Generic.Vector<float>) (rated:Generic.Vector<float>) (userId:int) (sim:similarity) =
    if unrated.[userId] > 0. then None // we have a rating already
    else
        let rating = rated.[userId]
        if rating = 0. then None // we have no rating to use
        else
            let overlap = nonZeroes unrated rated
            match overlap with
            | None -> None
            | Some(u, v) -> Some(rating, sim u v)

// compute weighted average of a sequence of (value, weight)
let weightedAverage (data: (float * float) seq) = 
    let weightedTotal, totalWeights = Seq.fold (fun (R,S) (r, s) -> (R + r * s, S + s)) (0., 0.) data
    if (totalWeights <= 0.) 
    then None 
    else Some(weightedTotal/totalWeights)
    
// Compute estimated rating for a dish not yet rated by user
let estimatedRating (data:Generic.Matrix<float>) (userId:int) (dishId:int) (sim:similarity) =
    let dish = data.Column(dishId)
    match (dish.[userId] > 0.) with
    | true -> None // dish has been rated already
    | false -> 
        let size = data.ColumnCount
        let ratings =
            [ 0 .. (size - 1) ]
            |> List.map (fun col -> (weightedRating dish (data.Column(col)) userId sim))
            |> List.choose id
        match ratings with
        | [] -> None
        | data -> weightedAverage data

let recommend (data:Generic.Matrix<float>) (userId:int) (sim:similarity) =
    let size = data.ColumnCount
    [ 0 .. (size - 1) ] 
    |> List.map (fun dishId -> dishId, estimatedRating data userId dishId sim)
    |> List.filter (fun (dishId, rating) -> rating.IsSome)
    |> List.map (fun (dishId, rating) -> (dishId, rating.Value))
    |> List.sortBy (fun (dishId, rating) -> - rating)
                        
let v0 = data.Column(0)
let v1 = data.Column(1)
let v2 = data.Column(2)

[ 0 .. 10 ]
|> List.iter (fun userId ->
    printfn "User %i" userId
    [ 0 .. 10 ]
    |> List.iter (fun dishId ->
        let r1 = estimatedRating data userId dishId euclideanSimilarity
        let r2 = estimatedRating data userId dishId cosineSimilarity
        printfn "... dish %i: eucl %A cos %A" dishId r1 r2)
)



// create synthetic data

let person1 = [| 5.; 4.; 1.; 1.; 2.; 1.; 1.; 5. |]
let person2 = [| 1.; 1.; 4.; 5.; 1.; 5.; 4.; 1. |]
let person3 = [| 4.; 5.; 4.; 1.; 1.; 2.; 2.; 1. |] 
let person4 = [| 1.; 1.; 1.; 1.; 1.; 1.; 1.; 5. |] 
let person5 = [| 1.; 1.; 5.; 4.; 5.; 1.; 1.; 1. |] 

let templates = [|
    person1;
    person2;
    person3;
    person4;
    person5 |]

let rng = new System.Random()
let density = 0.2

let createPerson (template:float[]) =
    template 
    |> Array.map (fun x -> if rng.NextDouble() < density then x else 0.)

let rows2 = 1000
let cols2 = 8

let syntheticData = DenseMatrix(rows2, cols2)

let ts = [| for row in 0 .. (rows2 - 1) -> rng.Next(0, 5) |]

for row in 0 .. (rows2 - 1) do
    let template = templates.[ts.[row]]
    let fake = createPerson template
    for col in 0 .. (cols2 - 1) do
        syntheticData.[row, col] <- fake.[col]

let svd2 = syntheticData.Svd(true)
//let U, sigmas, Vt = svd.U(), svd.S(), svd.VT()
