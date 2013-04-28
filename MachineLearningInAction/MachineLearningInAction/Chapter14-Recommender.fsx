// Chapter 14 covers SVD (Singular Value Decomposition),
// and how to use it for a Collaborative Recommendation Engine
 
#r @"..\..\MachineLearningInAction\packages\MathNet.Numerics.2.5.0\lib\net40\MathNet.Numerics.dll"
#r @"..\..\MachineLearningInAction\packages\MathNet.Numerics.FSharp.2.5.0\lib\net40\MathNet.Numerics.FSharp.dll"

open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.LinearAlgebra.Double
open MathNet.Numerics.Statistics

// Imagine we have users, with a database of movie ratings

type UserId = int
type MovieId = int
type Rating = { UserId:UserId; MovieId:MovieId; Rating:int }

// Let's imagine that we have 3 types of movies:
// Action, Romance, Documentary
// and 3 types of viewers, who only like 1 type of Movie.
// Let's imagine that 
// movies 0 to 3 are Action, 
// movies 4 to 7 are Romance, and 
// movies 8 to 11 are Documentary

// "User Templates":
//                Action .... Romance ... Documentary . 
let profile1 = [| 5; 4; 5; 4; 1; 1; 2; 1; 2; 1; 1; 1 |]
let profile2 = [| 1; 2; 1; 1; 4; 5; 5; 4; 1; 1; 1; 2 |]
let profile3 = [| 1; 1; 1; 2; 2; 2; 1; 1; 4; 5; 5; 5 |]
let profiles = [ profile1; profile2; profile3 ]

// Let's create a fake "synthetic dataset" from these 3 profiles
let rng = Random()
let proba = 0.4 // probability a movie was rated
// create fake ratings for a fake user,
// using the profile and id supplied
let createFrom profile userId =
    profile 
    |> Array.mapi (fun movieId rating -> 
        if rng.NextDouble() < proba 
        then Some({ UserId=userId; MovieId=movieId; Rating=rating }) 
        else None)
    |> Array.choose id

// Example: create ratings for user 42, who likes Romance
let romanceUser = createFrom profile2 42

// We generate 100 "fake" users and their ratings
let sampleSize = 100
let ratings = [
    for i in 0 .. (sampleSize - 1) do 
        yield! createFrom (profiles.[rng.Next(0, 3)]) i 
    ]

let movies = 12
let movieIds = [ 0 .. (movies - 1) ]

// We are set - let's start working
// and pull this sample into a data Matrix
let data = DenseMatrix(sampleSize, movies) 
for rating in ratings do
    data.[rating.UserId, rating.MovieId] <- Convert.ToDouble(rating.Rating)

// user 3 is now in row 3:
let fakeUser3 = data.Row(3)
printfn "User 3 ratings: %A" (fakeUser3 |> Seq.toList)

// To compute recommendations for a user
// we will need to know the rating (if it exists) 
// a user gave a movie, and
// how "similar" 2 movies are. 
type userRating = UserId -> MovieId -> float option
type movieSimilarity = MovieId -> MovieId -> float

/// Compute weighted average of a sequence of (value, weight)
let weightedAverage (data: (float * float) seq) = 
    let weightedTotal, totalWeights = 
        Seq.fold (fun (R,S) (r, s) -> 
            (R + r * s, S + s)) (0., 0.) data
    if (totalWeights <= 0.) 
    then None 
    else Some(weightedTotal/totalWeights)

/// Estimate the rating for an unrated movie:
/// by averaging known ratings, weighted by
/// how similar they are to the movie.
let estimate (similarity:movieSimilarity) 
             (rating:userRating) 
             (sample:MovieId seq) 
             (userId:UserId) 
             (movieId:MovieId) = 
    match (rating userId movieId) with
    | Some(_) -> None // already rated
    | None ->
        sample
        // for all rated movies, get rating
        // and similarity
        |> Seq.choose (fun id -> 
            let r = rating userId id
            match r with
            | None -> None
            | Some(value) -> Some(value, (similarity movieId id)))
        |> weightedAverage

/// Recommend movies: estimate rating for
/// all unrated movies and sort them 
/// by decreasing rating.
let recommend (similarity:movieSimilarity) 
              (rating:userRating) 
              (sample:MovieId seq) 
              (userId:UserId) =
    sample
    |> Seq.map (fun movieId -> 
        movieId, estimate similarity rating sample userId movieId)
    |> Seq.choose (fun (movieId, r) -> 
        match r with | None -> None | Some(value) -> Some(movieId, value))
    |> Seq.sortBy (fun (movieId, rating) -> - rating)
    |> Seq.toList

// Now let's build a naive recommender

// To make recommendations we need a similarity measure
type similarity = Generic.Vector<float> -> Generic.Vector<float> -> float

// Larger distances imply lower similarity
let euclideanSimilarity (v1: Generic.Vector<float>) (v2: Generic.Vector<float>) =
    1. / (1. + (v1 - v2).Norm(2.))

// Similarity based on the angle
let cosineSimilarity (v1: Generic.Vector<float>) v2 =
    v1.DotProduct(v2) / (v1.Norm(2.) * v2.Norm(2.))

// Similarity based on the Pearson correlation
let pearsonSimilarity (v1: Generic.Vector<float>) v2 =
    if v1.Count > 2 
    then 0.5 + 0.5 * Correlation.Pearson(v1, v2)
    else 1.

// Reduce 2 vectors to their non-zero pairs
let nonZeroes (v1:Generic.Vector<float>) 
              (v2:Generic.Vector<float>) =
    // Grab non-zero pairs of ratings 
    let size = v1.Count
    let overlap =
        [ 0 .. (size - 1) ] 
        |> List.fold (fun acc i -> 
            if v1.[i] > 0. && v2.[i] > 0. 
            then (v1.[i], v2.[i])::acc else acc) []
    // Recompose vectors if there is something left
    match overlap with
    | [] -> None
    | x  -> 
        let v1', v2' =
            x
            |> List.toArray
            |> Array.unzip
        Some(DenseVector(v1'), DenseVector(v2'))

// "Simple" similarity: keep only users that
// have rated both movies, and compare.
let simpleSimilarity (s:similarity) =
    fun (movie1:MovieId) (movie2:MovieId) ->
        let v1, v2 = data.Column(movie1), data.Column(movie2)
        let overlap = nonZeroes v1 v2
        match overlap with
        | None -> 0.
        | Some(v1', v2') -> s v1' v2'

// Return rating from data matrix, captured in closure
let simpleRating (userId:UserId) (movieId:MovieId) =
    let rating = data.[userId, movieId]
    if rating = 0. then None else Some(rating)

// Wire everything together: return a function
// that will produce a recommendation, based
// on whatever similarity function it is given
let simpleRecommender (s:similarity) =
    fun (userId:UserId) -> 
        recommend (simpleSimilarity s)
                  simpleRating
                  movieIds 
                  userId

// Illustration / usage
let simpleEuclidean = simpleRecommender euclideanSimilarity
let simpleCosine = simpleRecommender cosineSimilarity
let simplePearson = simpleRecommender pearsonSimilarity

let someUser = 42 // random user
let hisProfile = data.Row(someUser) |> Seq.toList
printfn "User ratings: %A" hisProfile
printfn "Simple recommendation"
printfn "Recommendation, Euclidean: %A" (simpleEuclidean someUser)
printfn "Recommendation, Cosine: %A" (simpleCosine someUser)
printfn "Recommendation, Pearson: %A" (simplePearson someUser)

// SVD based approach

// We'll retain only the largest values in the Sigma vector,
// (the diagonal of the S-matrix), which capture more than
// a given percentage of the "energy". 
let valuesForEnergy (min:float) (sigmas:Generic.Vector<float>) =
    let totalEnergy = sigmas.DotProduct(sigmas)
    let rec search i accEnergy =
        let x = sigmas.[i]
        let energy = x * x
        let percent = (accEnergy + energy)/totalEnergy
        match (percent >= min) with
        | true -> i
        | false -> search (i + 1) (accEnergy + energy)
    search 0 0.

let energy = 0.9 // arbitrary threshold

// Instead of comparing the columns / movies,
// we'll compare their projection using SVD,
// removing the less significant criteria
let data' =
    let svd = data.Svd(true)
    let U, sigmas = svd.U(), svd.S()
    let subset = valuesForEnergy energy sigmas
    let U' = U.SubMatrix(0, U.RowCount, 0, subset)
    let S' = DiagonalMatrix(subset, subset, sigmas.SubVector(0, (subset)).ToArray()) //S.SubMatrix(0, subset, 0, subset)

    (data.Transpose() * U' * S').Transpose()

// We can now compare similarity directly
// off the data' matrix computed by SVD
let svdSimilarity (s:similarity) =
    fun (movie1:MovieId) (movie2:MovieId) ->
        let v1, v2 = data'.Column(movie1), data'.Column(movie2)
        s v1 v2

// We can now create a recommender based off SVD similarity
let svdRecommender (s:similarity) =
    fun (userId:UserId) -> 
        recommend (svdSimilarity s)
                  simpleRating
                  movieIds 
                  userId

// Illustration, on same user profile as before
let svdEuclidean = svdRecommender euclideanSimilarity
let svdCosine = svdRecommender cosineSimilarity
let svdPearson = svdRecommender pearsonSimilarity

let sameUser = someUser
let sameProfile = data.Row(sameUser)
printfn "SVD-based recommendation"
printfn "Recommendation, Euclidean: %A" (svdEuclidean someUser)
printfn "Recommendation, Cosine: %A" (svdCosine someUser)
printfn "Recommendation, Pearson: %A" (svdPearson someUser)