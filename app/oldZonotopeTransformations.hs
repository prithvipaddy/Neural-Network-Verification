{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}
-- {-# LANGUAGE LiquidHaskell #-}

module Main where

-- import Data.Time.Clock
import Data.List ( transpose, maximumBy )
import Data.Aeson (FromJSON, decode)
import qualified Data.ByteString.Lazy as B (readFile)
import GHC.Generics (Generic)
import Data.Ord (comparing)
import Data.Maybe


type Vector' = [Double]
type Layer = ([Vector'],Vector', Bool) -- fst = list of list of weights, snd = list of bias, third = apply relu or not
type Zonotope = [[Double]]  -- [[3,1,0], [1,0.5,0.5]] = Zonotope: 3 + 1e1, 1 + 0.5e1 + 0.5e2

getCenter :: Zonotope -> Vector'
getCenter = map head

getGenerators :: Zonotope -> [Vector']
getGenerators z = Data.List.transpose (map tail z)

-- Rebuild a Zonotope from a center and list of generators
buildZonotope :: Vector' -> [Vector'] -> Zonotope
buildZonotope center generators = zipWith (:) center (Data.List.transpose generators)

vectorAdd :: Vector' -> Vector' -> Vector'
vectorAdd = zipWith (+)

scalarMult :: Double -> Vector' -> Vector'
scalarMult s = map (* s)

zonotopeAdd :: Zonotope -> Zonotope -> Zonotope
zonotopeAdd = zipWith (zipWith (+))

dotProduct :: Num a => [a] -> [a] -> a
dotProduct xs ys = sum (zipWith (*) xs ys)

-- abstract transformer for relu function
upper :: Zonotope -> [Double]
upper z =
    let
        c = getCenter z
        gs = getGenerators z
    in
        zipWith (+) c (map (sum . map abs) (Data.List.transpose gs))

lower :: Zonotope -> [Double]
lower z =
    let
        c = getCenter z
        gs = getGenerators z
    in
        zipWith (-) c (map (sum . map abs) (Data.List.transpose gs))

-- fst = lambda for each dimension of the zonotope, snd = n
findLambdaAndNRelu :: Zonotope -> ([Double],[Double])
findLambdaAndNRelu z =
    let
        u = upper z
        l = lower z
        lambda = zipWith (\ui li -> ui / (ui - li)) u l
        n = zipWith (\ui lambdai -> (ui * (1 - lambdai)) / 2) u lambda
    in
        (lambda,n)

scaleByLambdaRelu :: Zonotope -> Zonotope
scaleByLambdaRelu z =
    let
        lambda = fst (findLambdaAndNRelu z)
        c = getCenter z
        gs = getGenerators z
        scaledC = zipWith (*) lambda c
        scaledGs = map (zipWith (*) lambda) gs
    in
        buildZonotope scaledC scaledGs

-- CONSTRAINT - ZONOTOPE INPUT SHOULD BE 1D
composeLambdaAndNRelu :: Zonotope -> Zonotope
composeLambdaAndNRelu z =
    let
        lambdaScaled = scaleByLambdaRelu z
        lengthOfN = length (head z) + 1
        n = snd (findLambdaAndNRelu z)
    in
        zonotopeAdd (map (++ [0]) lambdaScaled) (nZonotopeForRelu lengthOfN (head n))

nZonotopeForRelu :: Int -> Double -> [[Double]]
nZonotopeForRelu i n | i <= 0 = []
            | i == 1 = [[n]]
            | otherwise = [[n] ++ replicate (i-2) 0 ++ [n]]

applyRelu :: Zonotope -> Zonotope
applyRelu = map (\row -> let singleZonotope = [row]
                         in head (composeLambdaAndNRelu singleZonotope))

-- Apply a linear transformation to a zonotope
linearTransform :: [Vector'] -> Vector' -> Zonotope -> Zonotope
linearTransform ws bias z =
    let
        c = getCenter z
        gs = getGenerators z
        -- Apply the matrix multiplication to the center and each generator
        transform v = map (sum . zipWith (*) v) ws
        newCenter = vectorAdd (transform c) bias
        newGenerators = map transform gs
    in
        buildZonotope newCenter newGenerators

applyNetwork :: [Layer] -> Zonotope -> Zonotope
applyNetwork layers z = foldl applyLayer z layers

applyLayer :: Zonotope -> Layer -> Zonotope
applyLayer acc (w, b, applyReLU) =
        let transformed = linearTransform w b acc
        in if applyReLU then applyRelu acc else transformed

-- Test data
testZ1 :: Zonotope
testZ1 = [[1, 0.5, 0], [1, 0, 0.5]]

hLayer1 :: Layer
hLayer1 = ([[2,0],[1,1]],[1,-1],False)

reluLayer :: Layer
reluLayer = ([[]],[],True)

outputLayer :: Layer
outputLayer = ([[1,-1]],[0.5],False)

loadLayers :: FilePath -> IO (Maybe [Layer])
loadLayers filePath = do
    content <- B.readFile filePath
    return (decode content)

testRelu1 :: Zonotope
testRelu1 = [[2,4]]  -- Zonotope: 2 + 4e1

testRelu2 :: Zonotope
testRelu2 = [[3,1,0], [1,0.5,0.5]]  -- Zonotope: 3 + 1e1, 1 + 0.5e1 + 0.5e2