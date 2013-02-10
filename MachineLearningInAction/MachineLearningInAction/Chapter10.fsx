// Chapter 10: K-means clustering
#load "KMeansClustering.fs"
open MachineLearning.KMeansClustering

// test on standard data
let rng = new System.Random()
let centroids = [ [| 0.; 0.; 0. |]; [| 20.; 30.; 40. |]; [| -40.; -50.; -60. |] ]
// Create 50 points centered around each Centroid
let data = [ 
    for centroid in centroids do
        for i in 1 .. 50 -> 
            Array.map (fun x -> x + 5. * (rng.NextDouble() - 0.5)) centroid ]

let factory = randomCentroids<float[]> rng
let search = kmeans euclidean factory avgCentroid data 3
let identifiedCentroids = fst search |> Seq.toList

// Just for quicks, I wondered if I could cluster strings

// Levenshtein distance between strings, lifted from:
// http://en.wikibooks.org/wiki/Algorithm_implementation/Strings/Levenshtein_distance#F.23
let inline min3 one two three = 
    if one < two && one < three then one
    elif two < three then two
    else three

let wagnerFischerLazy (s: string) (t: string) =
    let m = s.Length
    let n = t.Length
    let d = Array2D.create (m+1) (n+1) -1
    let rec dist =
        function
        | i, 0 -> i
        | 0, j -> j
        | i, j when d.[i,j] <> -1 -> d.[i,j]
        | i, j ->
            let dval = 
                if s.[i-1] = t.[j-1] then dist (i-1, j-1)
                else
                    min3
                        (dist (i-1, j)   + 1) // a deletion
                        (dist (i,   j-1) + 1) // an insertion
                        (dist (i-1, j-1) + 1) // a substitution
            d.[i, j] <- dval; dval 
    dist (m, n)

// Centroid update: pick the word in the cluster that
// has the lowest maximum distance to the others
let wordCentroid (current: string) (sample: string seq) =
    let size = Seq.length sample
    match size with
    | 0 -> current
    | _ ->
        sample
        |> Seq.map (fun word -> 
            let worst = 
                sample 
                |> Seq.map (fun s -> wagnerFischerLazy s word) 
                |> Seq.max
            word, worst)
        |> Seq.minBy snd
        |> fst

// Create a sample of words, based on three roots:
// http://www.learnthat.org/word_lists/view/12933
// http://www.learnthat.org/word_lists/view/13077
// http://www.learnthat.org/word_lists/view/12932
let words = [ 
    "AUTOGRAPH"; "GRAPHOLOGY"; "SEISMOGRAPH"; "TELEGRAPHIC"; "PARAGRAPH"; "CALLIGRAPHY"; "CRYPTOGRAPHY";
    "DESCRIBE"; "DESCRIPTION"; "SCRIBE"; "TYPESCRIPT"; "SCRIBBLE"; "TRANSCRIPTION"; "POSTSCRIPT";
    "ANAGRAM"; "CRYPTOGRAM"; "GRAMMAR"; "GRAMMARIAN"; "MONOGRAM"; "AEROGRAM"; "GRAMMATICAL" ]

let wordDistance w1 w2 = wagnerFischerLazy w1 w2 |> (float)
let wordFactory = randomCentroids<string> rng

let clusterize = kmeans wordDistance wordFactory wordCentroid words 3
let clusters = fst clusterize |> Seq.toArray
clusters |> Array.iter (fun w -> printfn "%s" w)
let assignment = snd clusterize |> Seq.toArray
words |> List.iteri (fun i w -> printfn "%s, %s, %f" w clusters.[fst assignment.[i]] (snd assignment.[i]))