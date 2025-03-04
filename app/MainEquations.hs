{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
-- {-# LANGUAGE LiquidHaskell #-}

module Main where

import Data.List (transpose)
import Data.Aeson (FromJSON, decode, encode)
import qualified Data.ByteString.Lazy as B (readFile, writeFile)
import GHC.Generics (Generic)
import Data.Vector (toList)
import Data.Matrix
import Data.Maybe (fromMaybe)
-- import System.Environment (getArgs)
import System.IO
import Control.Parallel.Strategies ( parMap, rpar, using, parListChunk )
-- import Data.Array.Unboxed (UArray, accumArray, elems)
-- import Data.Ix (range)

type Zonotope = Matrix Double

data LayerInfo = LayerInfo
  { layerName          :: String
  , layerType          :: String
  , inputShape         :: Maybe (Int,Int)
  , numFilters         :: Maybe Int
  , kernelSize         :: Maybe [Int]
  , activationFunction :: Maybe String
  , filters            :: Maybe [[[[Double]]]]
  , biases             :: Maybe [Double]
  , poolSize           :: Maybe [Int]
  , units              :: Maybe Int
  , weights            :: Maybe [[Double]]
  } deriving (Show, Generic)
instance FromJSON LayerInfo

data ImageData = ImageData
  { imageValues          :: [[[Double]]]
  , imageClass           :: Int
  , imageDimensions      :: (Int,Int,Int)
  } deriving (Show, Generic)
instance FromJSON ImageData

-- CONVOLUTION (from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8418593)

-- Function to generate one row of the weight matrix wF
generateRowWeightMatrixForConv :: [[Double]] -> (Int,Int) -> Int -> Int -> [Double]
generateRowWeightMatrixForConv kernel (inpImgRows,inpImgCols) i j =
  let kernelRows = length kernel
      kernelCols = length (head kernel)
      -- The if condition checks for a box the size of the kernel
      -- The position of the box is determined by i and j
      -- if i' and j' belong to the box then the relevent element from the kernel is selected
      row = [ if i' >= i && i' < i + kernelRows && j' >= j && j' < j + kernelCols
                then kernel !! (i' - i) !! (j' - j)
                else 0
            | i' <- [0 .. inpImgRows - 1]
            , j' <- [0 .. inpImgCols - 1]
            ]
   in row

-- TEST FOR GENERATE ROW
kGenerateRow :: [[Double]]
kGenerateRow = [[1,2,3],[4,5,6],[7,8,9]]

test1GenerateRow :: Bool
test1GenerateRow = let
  expected = [1.0,2.0,3.0,0.0,0.0,4.0,5.0,6.0,0.0,0.0,7.0,8.0,9.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
  result = generateRowWeightMatrixForConv kGenerateRow (5,5) 0 0
  in expected == result

-- Function to generate the full weight matrix wF
generateWeightMatrixForConv :: [[Double]] -> (Int,Int) -> [[Double]]
generateWeightMatrixForConv kernel (inpImgRows,inpImgCols) =
  let !kernelRows = length kernel
      !kernelCols = length (head kernel)
    -- Assuming that there is no padding and the stride is 1, we get the y dimensions below
    -- Here y signifies the output from convolution between the kernel and the input image
      !yRows = inpImgRows - kernelRows + 1
      !yCols = inpImgCols - kernelCols + 1
    -- One row of the Wf matrix transforms into one element of the y matrix
    -- i and j determine the row corresponding to element (i,j) from y
      !rows = parMap rpar (uncurry (generateRowWeightMatrixForConv kernel (inpImgRows, inpImgCols))) [(i, j) | i <- [0 .. yRows - 1], j <- [0 .. yCols - 1]]
   in rows

generateBiasMatrixForConv :: Matrix Double -> Double -> Matrix Double
generateBiasMatrixForConv z b = matrix (nrows z) (ncols z) (const b)

applySingleChannelConvolution :: Matrix Double -> (Int,Int) -> [[Double]] -> Matrix Double
applySingleChannelConvolution zonotope (inpImgRows,inpImgCols) kernel = let
    !wF = Data.Matrix.fromLists (generateWeightMatrixForConv kernel (inpImgRows,inpImgCols))
    !newZonotope = Data.Matrix.multStd wF zonotope
    in newZonotope

-- scalar composotion of the entire depth of the zonotopes to form a single zonotope
-- one scalar composition for each filter 
-- eg. if initial depth = 3, newZ is sum of the 3 zonotopes, and if numFilters = 32, new depth = 32
applyConvolutionPerFilter' :: [Matrix Double] -> (Int,Int) -> [[[Double]]] -> Matrix Double
applyConvolutionPerFilter' zonotope (inpImgRows,inpImgCols) kernel = let
    !convolutions = parMap rpar (\ (z1, k1) -> applySingleChannelConvolution z1 (inpImgRows, inpImgCols) k1) (zip zonotope kernel)
    !composedConvolutions = foldl1 (+) convolutions
    in composedConvolutions

applyConvolutionPerFilter :: [Matrix Double] -> (Int,Int) -> [[[Double]]] -> Double -> Matrix Double
applyConvolutionPerFilter zonotope (inpImgRows,inpImgCols) kernel bias = let
    !convolved = applyConvolutionPerFilter' zonotope (inpImgRows,inpImgCols) kernel
    !biasMatrix = generateBiasMatrixForConv convolved bias
    !final = convolved + biasMatrix
    in final

-- length of kernel and bias should be same
applyConvolution :: [Matrix Double] -> (Int,Int) -> [[[[Double]]]] -> [Double] -> [Matrix Double]
applyConvolution zonotope (inpImgRows,inpImgCols) kernel bias = let
  !convolutions = parMap rpar (uncurry (applyConvolutionPerFilter zonotope (inpImgRows, inpImgCols))) (zip kernel bias)
  in convolutions
--use par map above removing recursion

-- MAXPOOLING (from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8418593)
-- Generate the list of indices grouped for maxpooling

-- appendColumn :: Matrix a -> [a] -> Matrix a
-- appendColumn mat col
--   | nrows mat /= length col = error "Column length must match the number of rows in the matrix"
--   | otherwise =
--       let
--         !matRows = toLists mat
--         !newMatRows = zipWith (++) (map (:[]) col) matRows  -- Append each element of the column to each row
--       in fromLists newMatRows

-- generateWMPPooling :: Int -> Int -> [[Double]] -> [[Double]]
-- generateWMPPooling p q x =
--   let
--       xv = concat x
--       -- groupMap is a list of indices, which is used to reorder elements in xv
--       groupMap = concat (poolingGroups p q x)
--       sizeX = length xv
--       sizeG = length groupMap
--       -- Below, the i determines which row of wMP is being generated
--       -- j determines which position of the row will be occupied by '1'
--       !wmp = [[if i >= sizeG then 0.0 else if groupMap !! i == j then 1.0 else 0.0 | j <- [0 .. sizeX - 1]] | i <- [0 .. sizeX - 1]]
--    in wmp

{-
Example usage of generateWPooling
p = 2, q = 1
x:
[0,1,2,3]
[4,5,6,7]
[8,9,10,11]
[12,13,14,15]
xv = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

groups: [[0,4],[1,5],[2,6],[3,7],[8,12],[9,13],[10,14],[11,15]] 
NOTE FOR GROUPS: these will always be indices, it just so happens that x has been defined with values matching indices
groupMap = [0,4,1,5,2,6,3,7,8,12,9,13,10,14,11,15]

when i = 0, checks the value for groupMap at index 0.
since it is 0, it will input 1 into the first position of row 0.

i = 1, checks the value for groupMax at index 1.
passes through all j values upto 4, inserting '0' in those positions. at index 4, it will insert a '1'
this indicates that the 5th element from xv should be the 2nd element in xMP

Similarly done, upto i = 15, and wMP is constructed.
-}

{-
maxPooling :: Int -> Int -> [[Double]] -> [Double]
maxPooling p q x = let
    rows = length x
    cols = length (head x)
    newRows = rows `div` p
    newCols = cols `div` q
    wMP = Data.Matrix.transpose (Data.Matrix.fromLists (generateWMPPooling p q x))
    xFlat = Data.Matrix.fromLists [concat x]
    xMP' = multStd xFlat wMP
    xMP = Data.Matrix.toList xMP'
    sizeIdentity = length xMP
    size = length (concat (poolingGroups p q x))
    identity' = [[if i == j then 1 else 0 | j <- [0..sizeIdentity-1]] | i <- [0..sizeIdentity-1]]
    sublists = [take (p*q) (drop (i * p * q) xMP) | i <- [0..size `div` (p*q) - 1]]
    maxIndices = [startIdx + fst (maximumBy (comparing snd) (zip [0..] sublist)) | (startIdx, sublist) <- zip [0, p * q ..] sublists]
    selectedRows = Data.Matrix.fromLists [row | (idx, row) <- zip [0..] identity', idx `elem` maxIndices]
    maxPooledVector = Data.Matrix.toList (multStd xMP' (Data.Matrix.transpose selectedRows))
    maxPooledMatrix = Data.Matrix.toList (Data.Matrix.fromList newRows newCols maxPooledVector)
    in
        maxPooledMatrix

applySinglePointMaxPooling :: Int -> Int -> [Double] -> (Int,Int) -> [Double]
applySinglePointMaxPooling p q zonotope (rows,cols) = let
  x = Data.Matrix.toLists (Data.Matrix.fromList rows cols zonotope)
  in maxPooling p q x

applyAllPointsMaxPooling :: Int -> Int -> Matrix Double -> Int -> (Int,Int) -> Matrix Double
applyAllPointsMaxPooling p q _ 0 (imgRows,imgCols) = matrix ((imgRows `div` p)*(imgCols `div` q)) 0 (const 0)
applyAllPointsMaxPooling p q zonotope numPoints (imgRows,imgCols) = let
  point = Data.Vector.toList (getCol numPoints zonotope)
  maxPooledPoint = applySinglePointMaxPooling p q point (imgRows,imgCols)
  in appendColumn (applyAllPointsMaxPooling p q zonotope (numPoints - 1) (imgRows,imgCols)) maxPooledPoint

applyMaxPooling :: Int -> Int -> [Matrix Double] -> (Int,Int) -> [Matrix Double]
applyMaxPooling _ _ [] _ = []
applyMaxPooling p q (z:zonotope) (imgRows,imgCols) =  applyAllPointsMaxPooling p q z (ncols z) (imgRows,imgCols) : applyMaxPooling p q zonotope (imgRows,imgCols)
-}
-- AVERAGE POOLING
poolingGroups :: Int -> Int -> [[Double]] -> [[Int]]
poolingGroups p q x =
  let !rows = length x
      !cols = length (head x)
      -- groupIndices finds the indices for all elements in a pooling box.
      -- The input to the function is the top-left element of each box
      groupIndices i j = [ (i + di) * cols + (j + dj) | di <- [0 .. p - 1], dj <- [0 .. q - 1] ]
      -- Below, the groupIndices for the top-left element of every pooling group is found
      !groups = [ groupIndices (i * p) (j * q) | i <- [0 .. (rows `div` p) - 1], j <- [0 .. (cols `div` q) - 1] ]
   in groups

generateRowAveragePooling :: [Int] -> Int -> [Double]
generateRowAveragePooling positions size = let
  !row = [if i `elem` positions then 1.0 else 0.0 | i <- [0..size-1]]
  in row

averagePooling :: Int -> Int -> [[Double]] -> [Double]
averagePooling p q x = let
  !groups = poolingGroups p q x
  !sizeX = length x * length (head x)
  !poolSize' = fromIntegral p * fromIntegral q :: Double
  !wLists = [generateRowAveragePooling g sizeX | g <- groups]
  !w = Data.Matrix.fromLists wLists
  !zonotope = Data.Matrix.transpose (Data.Matrix.fromLists [concat x])
  in Data.Matrix.toList (Data.Matrix.scaleMatrix (1/poolSize') (multStd w zonotope))

applySinglePointAveragePooling :: Int -> Int -> [Double] -> (Int,Int) -> [Double]
applySinglePointAveragePooling p q zonotope (rows,cols) = let
  !x = Data.Matrix.toLists (Data.Matrix.fromList rows cols zonotope)
  !pooled = averagePooling p q x
  in pooled

applyAllPointsAveragePooling :: Int -> Int -> Matrix Double -> (Int,Int) -> Matrix Double
applyAllPointsAveragePooling p q zonotope (imgRows,imgCols) = let
  !allPoints = map (\num -> Data.Vector.toList (getCol num zonotope)) [1..(ncols zonotope)]
  !avgPooledPoints = map (\point -> applySinglePointAveragePooling p q point (imgRows,imgCols)) allPoints `using` parListChunk 100 rpar
  !result = Data.Matrix.transpose (fromLists avgPooledPoints)
  in result

applyAveragePooling :: Int -> Int -> [Matrix Double] -> (Int,Int) -> [Matrix Double]
applyAveragePooling p q zonotope (imgRows,imgCols) = let
  !avgPooled = parMap rpar (\z -> applyAllPointsAveragePooling p q z (imgRows,imgCols)) zonotope
  in avgPooled

-- AVERAGE POOLING TESTS
{-
x = [1,2,3,4]
    [5,6,7,8]
(2x2) x -> [14/4, 22/4]
(2x1) x -> [3,4,5,6]
(1x2) x -> [1.5,3.5]
           [5.5,7.5]
-}
x1AveragePooling :: Matrix Double
x1AveragePooling = Data.Matrix.transpose (Data.Matrix.fromLists [[1,2,3,4,5,6,7,8]])

test1AveragePooling :: Bool
test1AveragePooling = let
  pooled = applyAveragePooling 2 2 [x1AveragePooling] (2,4)
  expected = [fromLists [[3.5],[5.5]]]
  in pooled==expected

-- RELU (SHEARING)
reluUpper :: [Double] -> Double
reluUpper eq = let
  center = head eq
  generators = tail eq
  in center + sum (map abs generators)

reluLower :: [Double] -> Double
reluLower eq = let
  center = head eq
  generators = tail eq
  in center - sum (map abs generators)

composeLambdaAndNRelu :: [Double] -> [Double]
composeLambdaAndNRelu eq = let
    !u = reluUpper eq
    !l = reluLower eq
    !lambda = u / (u-l)
    !n = (u * (1 - lambda)) / 2
    !lambdaScaled = map (* lambda) eq ++ [0]
    !lengthLambdaScaled = length lambdaScaled
    !nScaled = [n] ++ replicate (lengthLambdaScaled - 2) 0 ++ [n]
    in zipWith (+) lambdaScaled nScaled

applyReluPerDimension :: [Double] -> [Double]
applyReluPerDimension eq = let
  !l = reluLower eq
  !u = reluUpper eq
  !eqLength = length eq
  in if l > 0
    then eq ++ [0]
    else if u < 0
      then replicate (eqLength + 1) 0
    else
      composeLambdaAndNRelu eq

applyRelu :: [[[Double]]] -> [[[Double]]]
applyRelu zonotope = let
  !result = map (map applyReluPerDimension) zonotope
  in result

-- DENSE
-- weights x zonotope
applyDenseLayerWeights :: Matrix Double -> Matrix Double -> Matrix Double
applyDenseLayerWeights = multStd

--TESTING APPLYING CONV AND MAXPOOLING TO A TEST ZONOTOPE
--  dimensions = 3 x 16 x 5
testZ1 :: [Matrix Double]
testZ1 = replicate 3 (Data.Matrix.fromLists (replicate 16 [1,2,-3,4,-5]))

-- dimensions = 3 x 1849 x 4
testZ2 :: [Matrix Double]
testZ2 = replicate 3 (Data.Matrix.fromLists (replicate 1849 [0.8,0.9,1,1,1.1]))

readLayers :: String -> IO [LayerInfo]
readLayers filepath = do
  !jsonData <- B.readFile filepath
  let !layers = fromMaybe [] (decode jsonData :: Maybe [LayerInfo])
  return layers

processLayers :: Handle -> String -> [Matrix Double] -> IO [Matrix Double]
processLayers logFile filepath zonotope = do
  !layers <- readLayers filepath
  case layers of
    [] -> do
      hPutStrLn logFile "Layers not found"
      return []
    layers' -> do
      let !inputShape' = fromMaybe (-1,-1) (inputShape (head layers'))
      case inputShape' of
        (-1,-1) -> do
          hPutStrLn logFile "Input shape for the neural network not found"
          return []
        inputShape'' -> do
          !finalZonotope <- parseLayers logFile (tail layers') zonotope inputShape''
          hPutStrLn logFile $ "Final zonotope dimensions: " ++ show (length finalZonotope,nrows (head finalZonotope),ncols (head finalZonotope))
          print finalZonotope
          return finalZonotope

{-
1 unique generator per dimension:
If zonotope is 3D and its center is (2,3,0.5), then there will be generators e1, e2 and e3 acting as follows:
2 + 1 e1
3 +       1 e2
0.5 +           1 e3
i.e
[[2,1,0,0],
 [3,0,1,0],
 [0.5,0,0,1]]
-}
convertImageDataToSingleZonotopePoint :: String -> IO ([Matrix Double],Int)
convertImageDataToSingleZonotopePoint filepath = do
  !jsonData <- B.readFile filepath
  let !image = fromMaybe [] (decode jsonData :: Maybe [ImageData])
  case image of
    [] -> do
      putStrLn "Image data not found"
      return ([],-1)
    images' -> do
      let
        !img1 = imageValues (head images')
        !(height1,width1,_) = imageDimensions (head images')
        !concatImg = Data.List.transpose (concat img1)
        !img1Zonotope = map (Data.Matrix.fromList (height1 * width1) 1) concatImg
      let !img1Class = imageClass (head images')
      return (img1Zonotope,img1Class)

parseLayers :: Handle -> [LayerInfo] -> [Matrix Double] -> (Int,Int) -> IO [Matrix Double]
parseLayers _ [] zonotope _ = return zonotope
parseLayers logFile (l:layers) zonotope (imgRows,imgCols) = do
  -- Print the layer type (name) of the current layer
  hPutStrLn logFile $ "parsed layer: " ++ layerName l
  hPutStrLn logFile $ "img dimensions: " ++ show (imgRows,imgCols)
  hPutStrLn logFile $ "zonotope dimensions: " ++ show (length zonotope,nrows (head zonotope),ncols (head zonotope))
  case layerType l of
    "<class 'keras.src.layers.convolutional.conv2d.Conv2D'>" ->
      let
        !kernelSize' = fromMaybe [] (kernelSize l)
        !newRows = imgRows - head kernelSize' + 1
        !newCols = imgCols - head (tail kernelSize') + 1
        !newZ' = applyConvolution zonotope (imgRows,imgCols) (fromMaybe [] (filters l)) (fromMaybe [] (biases l))
        !activation = fromMaybe [] (activationFunction l)
        !newZ = if activation == "relu"
          then map Data.Matrix.fromLists (applyRelu (map Data.Matrix.toLists newZ'))
          else newZ'
      in parseLayers logFile layers newZ (newRows,newCols)
    -- "<class 'keras.src.layers.pooling.max_pooling2d.MaxPooling2D'>" ->
    --   let poolSize' = fromMaybe [] (poolSize l)
    --       p = head poolSize'
    --       q = head (tail poolSize')
    --       newRows = imgRows `div` p
    --       newCols = imgCols `div` q
    --   in parseLayers logFile layers (applyMaxPooling p q zonotope (imgRows,imgCols)) (newRows,newCols)
    "<class 'keras.src.layers.pooling.average_pooling2d.AveragePooling2D'>" ->
      let !poolSize' = fromMaybe [] (poolSize l)
          !p = head poolSize'
          !q = head (tail poolSize')
          !newRows = imgRows `div` p
          !newCols = imgCols `div` q
          !newZ = parseLayers logFile layers (applyAveragePooling p q zonotope (imgRows,imgCols)) (newRows,newCols)
      in newZ
    "<class 'keras.src.layers.reshaping.flatten.Flatten'>" ->
      let
        !newRows = 1
        !newCols = imgRows * imgCols
        !newZ = parseLayers logFile layers [Data.Matrix.fromList (length zonotope) (ncols (head zonotope)) (concatMap Data.Matrix.toList zonotope)] (newRows,newCols)
      in newZ
    "<class 'keras.src.layers.core.dense.Dense'>" ->
      let !weights' = fromMaybe [] (weights l)
          !weightsMatrix = Data.Matrix.transpose (Data.Matrix.fromLists weights')
          !zonotopeMatrix = head zonotope
          !newZ' = [applyDenseLayerWeights weightsMatrix zonotopeMatrix]
          !newZ2 = Data.Matrix.toLists (head newZ')
          !biases' = fromMaybe [] (biases l)
          !newZ'' = Data.Matrix.fromLists (zipWith (\x row -> (head row + x) : tail row) biases' newZ2)
          !activationFunction' = fromMaybe [] (activationFunction l)
          !newZ = if activationFunction' == "relu"
            then map Data.Matrix.fromLists (applyRelu [Data.Matrix.toLists newZ''])
            else [newZ'']
      in parseLayers logFile layers newZ (imgRows,imgCols)
    "<class 'keras.src.layers.regularization.dropout.Dropout'>" ->
      parseLayers logFile layers zonotope (imgRows,imgCols)
    _ ->
      return []

-- CHECKING IF THE ARGMAX VALUE IN ALL POINTS MATCHES

checkArgMax :: [(Double,Double)] -> Int -> Bool
checkArgMax zonotope label = let
  !(lowerBoundForExpectedLabel,_) = zonotope !! label
  !otherTuples = take label zonotope ++ drop (label + 1) zonotope
  in all (\(_, y) -> y < lowerBoundForExpectedLabel) otherTuples

-- CREATING EQUATIONS FOR EACH DIMENSION (1 + 0 E1 + 1 E2 BECOMES [1,0,1])
{-
let zonotope = [1,2,3]
then equations = [(1 + 1 e1),
                  (2 +       1 e2),
                  (3 +             1 e3)]
i.e. equations = [1,1,0,0
                  2,0,1,0
                  3,0,0,1]
-}
createEquations :: Matrix Double -> Double -> Matrix Double
createEquations zonotope perturbation = let
  !size = nrows zonotope
  !identity' = Data.Matrix.identity size :: Matrix Double
  !perturbedIdentity = scaleMatrix perturbation identity'
  !newZ = zonotope <|> perturbedIdentity
  in newZ

solveEquations :: Matrix Double -> [(Double,Double)]
solveEquations zonotope = let
  !zonotopeLists = toLists zonotope
  !solved = map findBoundsPerDimension zonotopeLists
  in solved

findBoundsPerDimension :: [Double] -> (Double,Double)
findBoundsPerDimension equation = let
  !center = head equation
  !generators = tail equation
  !upperBound' = center + sum (map abs generators)
  !lowerBound' = center - sum (map abs generators)
  in (lowerBound',upperBound')

-- MAIN WITH ONLY A SINGLE POINT IN THE ZONOTOPE 
main :: IO ([[(Double,Double)]],[Bool])
main = do
  logFile <- openFile "NeuralNetVerification_output.log" AppendMode
  hSetBuffering logFile NoBuffering
  (zonotope,correctLabel) <- convertImageDataToSingleZonotopePoint "/data_home/Prithvi/haskell/app/imageData.json"
  finalZonotopeEquations <- processLayers logFile "/data_home/Prithvi/haskell/app/layersInfo.json" zonotope
  let finalBounds = map solveEquations finalZonotopeEquations
  let correctlyClassified = map (`checkArgMax` correctLabel) finalBounds
  hPutStrLn logFile $ "Final zonotpe" ++ show correctlyClassified
  let jsonData2 = encode finalBounds
  B.writeFile "correctlyClassified.json" jsonData2
  hPutStrLn logFile "Data written to correctlyClassified.json"
  hClose logFile
  return (finalBounds,correctlyClassified)

-- MAIN WITH ZONOTOPE WITH ALL EQUATIONS (ONE UNIQUE GENERATOR PER DIMENSION)
-- main :: IO ([[(Double,Double)]],[Bool])
-- main = do
--   !logFile <- openFile "neuralNetVerification_output.log" AppendMode
--   hSetBuffering logFile NoBuffering
--   !args <- getArgs
--   let !perturbation = read (head args) :: Double
--   !(zonotope,correctLabel) <- convertImageDataToSingleZonotopePoint "/data_home/Prithvi/haskell/app/imageData.json"
--   let
--     !perturbedZonotope = map (`createEquations` perturbation) zonotope
--   !finalZonotopeEquations <- processLayers logFile "/data_home/Prithvi/haskell/app/layersInfo.json" perturbedZonotope
--   let !finalBounds = map solveEquations finalZonotopeEquations
--   let !correctlyClassified = map (`checkArgMax` correctLabel) finalBounds
--   hPutStrLn logFile $ "Final zonotpe" ++ show correctlyClassified
--   let jsonData = encode correctlyClassified
--   B.writeFile "correctlyClassified.json" jsonData
--   hPutStrLn logFile "Data written to correctlyClassified.json"
--   hClose logFile
--   return (finalBounds,correctlyClassified)


-- "/Users/prithvi/Documents/Krea/Capstone/AbstractVerification/Zonotope/haskell/app/imageData.json"