from nose.tools import with_setup #, classmethod
from StringIO import StringIO
import corenlp
from discourse.models.ngram import NGramDiscourseInstance
from discourse.lattice import Transition

class TestNGramModel:

    @classmethod
    def setup_class(cls):

        xml_str = """<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet href="CoreNLP-to-HTML.xsl" type="text/xsl"?>
<root>
  <document>
    <sentences>
      <sentence id="1" sentimentValue="1" sentiment="Negative">
        <tokens>
          <token id="1">
            <word>BC-China-Earthquake</word>
            <lemma>bc-china-earthquake</lemma>
            <CharacterOffsetBegin>0</CharacterOffsetBegin>
            <CharacterOffsetEnd>19</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="2">
            <word>,</word>
            <lemma>,</lemma>
            <CharacterOffsetBegin>19</CharacterOffsetBegin>
            <CharacterOffsetEnd>20</CharacterOffsetEnd>
            <POS>,</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="3">
            <word>1st</word>
            <lemma>1st</lemma>
            <CharacterOffsetBegin>21</CharacterOffsetBegin>
            <CharacterOffsetEnd>24</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>ORDINAL</NER>
            <NormalizedNER>1.0</NormalizedNER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="4">
            <word>Ld-Writethru</word>
            <lemma>Ld-Writethru</lemma>
            <CharacterOffsetBegin>25</CharacterOffsetBegin>
            <CharacterOffsetEnd>37</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="5">
            <word>|</word>
            <lemma>|</lemma>
            <CharacterOffsetBegin>37</CharacterOffsetBegin>
            <CharacterOffsetEnd>38</CharacterOffsetEnd>
            <POS>CD</POS>
            <NER>NUMBER</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="6">
            <word>URGENT</word>
            <lemma>URGENT</lemma>
            <CharacterOffsetBegin>38</CharacterOffsetBegin>
            <CharacterOffsetEnd>44</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="7">
            <word>|</word>
            <lemma>|</lemma>
            <CharacterOffsetBegin>44</CharacterOffsetBegin>
            <CharacterOffsetEnd>45</CharacterOffsetEnd>
            <POS>CD</POS>
            <NER>NUMBER</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="8">
            <word>Earthquake</word>
            <lemma>earthquake</lemma>
            <CharacterOffsetBegin>45</CharacterOffsetBegin>
            <CharacterOffsetEnd>55</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="9">
            <word>Rocks</word>
            <lemma>rock</lemma>
            <CharacterOffsetBegin>56</CharacterOffsetBegin>
            <CharacterOffsetEnd>61</CharacterOffsetEnd>
            <POS>NNS</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="10">
            <word>Northwestern</word>
            <lemma>Northwestern</lemma>
            <CharacterOffsetBegin>62</CharacterOffsetBegin>
            <CharacterOffsetEnd>74</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="11">
            <word>Xinjiang</word>
            <lemma>Xinjiang</lemma>
            <CharacterOffsetBegin>75</CharacterOffsetBegin>
            <CharacterOffsetEnd>83</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>LOCATION</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="12">
            <word>Mountains</word>
            <lemma>mountain</lemma>
            <CharacterOffsetBegin>84</CharacterOffsetBegin>
            <CharacterOffsetEnd>93</CharacterOffsetEnd>
            <POS>NNS</POS>
            <NER>LOCATION</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="13">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>93</CharacterOffsetBegin>
            <CharacterOffsetEnd>94</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
        </tokens>
        <parse>(ROOT (NP (NP (JJ BC-China-Earthquake)) (, ,) (NP (NP (JJ 1st) (NNP Ld-Writethru)) (NP (CD |) (NNP URGENT) (CD |) (NN Earthquake) (NNS Rocks)) (NP (NNP Northwestern) (NNP Xinjiang) (NNS Mountains))) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="1">BC-China-Earthquake</dependent>
          </dep>
          <dep type="amod">
            <governor idx="4">Ld-Writethru</governor>
            <dependent idx="3">1st</dependent>
          </dep>
          <dep type="appos">
            <governor idx="1">BC-China-Earthquake</governor>
            <dependent idx="4">Ld-Writethru</dependent>
          </dep>
          <dep type="num">
            <governor idx="9">Rocks</governor>
            <dependent idx="5">|</dependent>
          </dep>
          <dep type="nn">
            <governor idx="9">Rocks</governor>
            <dependent idx="6">URGENT</dependent>
          </dep>
          <dep type="num">
            <governor idx="9">Rocks</governor>
            <dependent idx="7">|</dependent>
          </dep>
          <dep type="nn">
            <governor idx="9">Rocks</governor>
            <dependent idx="8">Earthquake</dependent>
          </dep>
          <dep type="dep">
            <governor idx="4">Ld-Writethru</governor>
            <dependent idx="9">Rocks</dependent>
          </dep>
          <dep type="nn">
            <governor idx="12">Mountains</governor>
            <dependent idx="10">Northwestern</dependent>
          </dep>
          <dep type="nn">
            <governor idx="12">Mountains</governor>
            <dependent idx="11">Xinjiang</dependent>
          </dep>
          <dep type="dep">
            <governor idx="4">Ld-Writethru</governor>
            <dependent idx="12">Mountains</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="1">BC-China-Earthquake</dependent>
          </dep>
          <dep type="amod">
            <governor idx="4">Ld-Writethru</governor>
            <dependent idx="3">1st</dependent>
          </dep>
          <dep type="appos">
            <governor idx="1">BC-China-Earthquake</governor>
            <dependent idx="4">Ld-Writethru</dependent>
          </dep>
          <dep type="num">
            <governor idx="9">Rocks</governor>
            <dependent idx="5">|</dependent>
          </dep>
          <dep type="nn">
            <governor idx="9">Rocks</governor>
            <dependent idx="6">URGENT</dependent>
          </dep>
          <dep type="num">
            <governor idx="9">Rocks</governor>
            <dependent idx="7">|</dependent>
          </dep>
          <dep type="nn">
            <governor idx="9">Rocks</governor>
            <dependent idx="8">Earthquake</dependent>
          </dep>
          <dep type="dep">
            <governor idx="4">Ld-Writethru</governor>
            <dependent idx="9">Rocks</dependent>
          </dep>
          <dep type="nn">
            <governor idx="12">Mountains</governor>
            <dependent idx="10">Northwestern</dependent>
          </dep>
          <dep type="nn">
            <governor idx="12">Mountains</governor>
            <dependent idx="11">Xinjiang</dependent>
          </dep>
          <dep type="dep">
            <governor idx="4">Ld-Writethru</governor>
            <dependent idx="12">Mountains</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="1">BC-China-Earthquake</dependent>
          </dep>
          <dep type="amod">
            <governor idx="4">Ld-Writethru</governor>
            <dependent idx="3">1st</dependent>
          </dep>
          <dep type="appos">
            <governor idx="1">BC-China-Earthquake</governor>
            <dependent idx="4">Ld-Writethru</dependent>
          </dep>
          <dep type="num">
            <governor idx="9">Rocks</governor>
            <dependent idx="5">|</dependent>
          </dep>
          <dep type="nn">
            <governor idx="9">Rocks</governor>
            <dependent idx="6">URGENT</dependent>
          </dep>
          <dep type="num">
            <governor idx="9">Rocks</governor>
            <dependent idx="7">|</dependent>
          </dep>
          <dep type="nn">
            <governor idx="9">Rocks</governor>
            <dependent idx="8">Earthquake</dependent>
          </dep>
          <dep type="dep">
            <governor idx="4">Ld-Writethru</governor>
            <dependent idx="9">Rocks</dependent>
          </dep>
          <dep type="nn">
            <governor idx="12">Mountains</governor>
            <dependent idx="10">Northwestern</dependent>
          </dep>
          <dep type="nn">
            <governor idx="12">Mountains</governor>
            <dependent idx="11">Xinjiang</dependent>
          </dep>
          <dep type="dep">
            <governor idx="4">Ld-Writethru</governor>
            <dependent idx="12">Mountains</dependent>
          </dep>
        </dependencies>
      </sentence>
      <sentence id="2" sentimentValue="1" sentiment="Negative">
        <tokens>
          <token id="1">
            <word>BEIJING</word>
            <lemma>BEIJING</lemma>
            <CharacterOffsetBegin>96</CharacterOffsetBegin>
            <CharacterOffsetEnd>103</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>LOCATION</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="2">
            <word>-LRB-</word>
            <lemma>-lrb-</lemma>
            <CharacterOffsetBegin>104</CharacterOffsetBegin>
            <CharacterOffsetEnd>105</CharacterOffsetEnd>
            <POS>-LRB-</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="3">
            <word>AP</word>
            <lemma>AP</lemma>
            <CharacterOffsetBegin>105</CharacterOffsetBegin>
            <CharacterOffsetEnd>107</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>ORGANIZATION</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="4">
            <word>-RRB-</word>
            <lemma>-rrb-</lemma>
            <CharacterOffsetBegin>107</CharacterOffsetBegin>
            <CharacterOffsetEnd>108</CharacterOffsetEnd>
            <POS>-RRB-</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="5">
            <word>A</word>
            <lemma>a</lemma>
            <CharacterOffsetBegin>109</CharacterOffsetBegin>
            <CharacterOffsetEnd>110</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="6">
            <word>strong</word>
            <lemma>strong</lemma>
            <CharacterOffsetBegin>111</CharacterOffsetBegin>
            <CharacterOffsetEnd>117</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="7">
            <word>earthquake</word>
            <lemma>earthquake</lemma>
            <CharacterOffsetBegin>118</CharacterOffsetBegin>
            <CharacterOffsetEnd>128</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="8">
            <word>hit</word>
            <lemma>hit</lemma>
            <CharacterOffsetBegin>129</CharacterOffsetBegin>
            <CharacterOffsetEnd>132</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="9">
            <word>a</word>
            <lemma>a</lemma>
            <CharacterOffsetBegin>133</CharacterOffsetBegin>
            <CharacterOffsetEnd>134</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="10">
            <word>remote</word>
            <lemma>remote</lemma>
            <CharacterOffsetBegin>135</CharacterOffsetBegin>
            <CharacterOffsetEnd>141</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="11">
            <word>area</word>
            <lemma>area</lemma>
            <CharacterOffsetBegin>142</CharacterOffsetBegin>
            <CharacterOffsetEnd>146</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="12">
            <word>in</word>
            <lemma>in</lemma>
            <CharacterOffsetBegin>147</CharacterOffsetBegin>
            <CharacterOffsetEnd>149</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="13">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>150</CharacterOffsetBegin>
            <CharacterOffsetEnd>153</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="14">
            <word>Altai</word>
            <lemma>Altai</lemma>
            <CharacterOffsetBegin>154</CharacterOffsetBegin>
            <CharacterOffsetEnd>159</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>LOCATION</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="15">
            <word>mountains</word>
            <lemma>mountain</lemma>
            <CharacterOffsetBegin>160</CharacterOffsetBegin>
            <CharacterOffsetEnd>169</CharacterOffsetEnd>
            <POS>NNS</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="16">
            <word>in</word>
            <lemma>in</lemma>
            <CharacterOffsetBegin>170</CharacterOffsetBegin>
            <CharacterOffsetEnd>172</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="17">
            <word>northwestern</word>
            <lemma>northwestern</lemma>
            <CharacterOffsetBegin>173</CharacterOffsetBegin>
            <CharacterOffsetEnd>185</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="18">
            <word>Xinjiang</word>
            <lemma>Xinjiang</lemma>
            <CharacterOffsetBegin>186</CharacterOffsetBegin>
            <CharacterOffsetEnd>194</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>LOCATION</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="19">
            <word>early</word>
            <lemma>early</lemma>
            <CharacterOffsetBegin>195</CharacterOffsetBegin>
            <CharacterOffsetEnd>200</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>DATE</NER>
            <NormalizedNER>XXXX-WXX-3</NormalizedNER>
            <Speaker>PER0</Speaker>
            <Timex tid="t1" type="DATE">XXXX-WXX-3</Timex>
          </token>
          <token id="20">
            <word>Wednesday</word>
            <lemma>Wednesday</lemma>
            <CharacterOffsetBegin>201</CharacterOffsetBegin>
            <CharacterOffsetEnd>210</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>DATE</NER>
            <NormalizedNER>XXXX-WXX-3</NormalizedNER>
            <Speaker>PER0</Speaker>
            <Timex tid="t1" type="DATE">XXXX-WXX-3</Timex>
          </token>
          <token id="21">
            <word>,</word>
            <lemma>,</lemma>
            <CharacterOffsetBegin>210</CharacterOffsetBegin>
            <CharacterOffsetEnd>211</CharacterOffsetEnd>
            <POS>,</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="22">
            <word>according</word>
            <lemma>accord</lemma>
            <CharacterOffsetBegin>212</CharacterOffsetBegin>
            <CharacterOffsetEnd>221</CharacterOffsetEnd>
            <POS>VBG</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="23">
            <word>to</word>
            <lemma>to</lemma>
            <CharacterOffsetBegin>222</CharacterOffsetBegin>
            <CharacterOffsetEnd>224</CharacterOffsetEnd>
            <POS>TO</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="24">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>225</CharacterOffsetBegin>
            <CharacterOffsetEnd>228</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="25">
            <word>Central</word>
            <lemma>Central</lemma>
            <CharacterOffsetBegin>229</CharacterOffsetBegin>
            <CharacterOffsetEnd>236</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>ORGANIZATION</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="26">
            <word>Seismology</word>
            <lemma>Seismology</lemma>
            <CharacterOffsetBegin>237</CharacterOffsetBegin>
            <CharacterOffsetEnd>247</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>ORGANIZATION</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="27">
            <word>Bureau</word>
            <lemma>Bureau</lemma>
            <CharacterOffsetBegin>248</CharacterOffsetBegin>
            <CharacterOffsetEnd>254</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>ORGANIZATION</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="28">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>254</CharacterOffsetBegin>
            <CharacterOffsetEnd>255</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
        </tokens>
        <parse>(ROOT (S (NP (NNP BEIJING)) (PRN (-LRB- -LRB-) (NP (NNP AP)) (-RRB- -RRB-)) (NP (DT A) (JJ strong) (NN earthquake)) (VP (VBD hit) (NP (DT a) (JJ remote) (NN area)) (PP (IN in) (NP (DT the) (NNP Altai) (NNS mountains))) (PP (IN in) (NP (JJ northwestern) (NNP Xinjiang))) (NP-TMP (JJ early) (NNP Wednesday)) (, ,) (PP (VBG according) (PP (TO to) (NP (DT the) (NNP Central) (NNP Seismology) (NNP Bureau))))) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="8">hit</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="8">hit</governor>
            <dependent idx="1">BEIJING</dependent>
          </dep>
          <dep type="dep">
            <governor idx="8">hit</governor>
            <dependent idx="3">AP</dependent>
          </dep>
          <dep type="det">
            <governor idx="7">earthquake</governor>
            <dependent idx="5">A</dependent>
          </dep>
          <dep type="amod">
            <governor idx="7">earthquake</governor>
            <dependent idx="6">strong</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="8">hit</governor>
            <dependent idx="7">earthquake</dependent>
          </dep>
          <dep type="det">
            <governor idx="11">area</governor>
            <dependent idx="9">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="11">area</governor>
            <dependent idx="10">remote</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="8">hit</governor>
            <dependent idx="11">area</dependent>
          </dep>
          <dep type="prep">
            <governor idx="8">hit</governor>
            <dependent idx="12">in</dependent>
          </dep>
          <dep type="det">
            <governor idx="15">mountains</governor>
            <dependent idx="13">the</dependent>
          </dep>
          <dep type="nn">
            <governor idx="15">mountains</governor>
            <dependent idx="14">Altai</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="12">in</governor>
            <dependent idx="15">mountains</dependent>
          </dep>
          <dep type="prep">
            <governor idx="8">hit</governor>
            <dependent idx="16">in</dependent>
          </dep>
          <dep type="amod">
            <governor idx="18">Xinjiang</governor>
            <dependent idx="17">northwestern</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="16">in</governor>
            <dependent idx="18">Xinjiang</dependent>
          </dep>
          <dep type="amod">
            <governor idx="20">Wednesday</governor>
            <dependent idx="19">early</dependent>
          </dep>
          <dep type="tmod">
            <governor idx="8">hit</governor>
            <dependent idx="20">Wednesday</dependent>
          </dep>
          <dep type="prep">
            <governor idx="8">hit</governor>
            <dependent idx="22">according</dependent>
          </dep>
          <dep type="pcomp">
            <governor idx="22">according</governor>
            <dependent idx="23">to</dependent>
          </dep>
          <dep type="det">
            <governor idx="27">Bureau</governor>
            <dependent idx="24">the</dependent>
          </dep>
          <dep type="nn">
            <governor idx="27">Bureau</governor>
            <dependent idx="25">Central</dependent>
          </dep>
          <dep type="nn">
            <governor idx="27">Bureau</governor>
            <dependent idx="26">Seismology</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="23">to</governor>
            <dependent idx="27">Bureau</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="8">hit</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="8">hit</governor>
            <dependent idx="1">BEIJING</dependent>
          </dep>
          <dep type="dep">
            <governor idx="8">hit</governor>
            <dependent idx="3">AP</dependent>
          </dep>
          <dep type="det">
            <governor idx="7">earthquake</governor>
            <dependent idx="5">A</dependent>
          </dep>
          <dep type="amod">
            <governor idx="7">earthquake</governor>
            <dependent idx="6">strong</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="8">hit</governor>
            <dependent idx="7">earthquake</dependent>
          </dep>
          <dep type="det">
            <governor idx="11">area</governor>
            <dependent idx="9">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="11">area</governor>
            <dependent idx="10">remote</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="8">hit</governor>
            <dependent idx="11">area</dependent>
          </dep>
          <dep type="det">
            <governor idx="15">mountains</governor>
            <dependent idx="13">the</dependent>
          </dep>
          <dep type="nn">
            <governor idx="15">mountains</governor>
            <dependent idx="14">Altai</dependent>
          </dep>
          <dep type="prep_in">
            <governor idx="8">hit</governor>
            <dependent idx="15">mountains</dependent>
          </dep>
          <dep type="amod">
            <governor idx="18">Xinjiang</governor>
            <dependent idx="17">northwestern</dependent>
          </dep>
          <dep type="prep_in">
            <governor idx="8">hit</governor>
            <dependent idx="18">Xinjiang</dependent>
          </dep>
          <dep type="amod">
            <governor idx="20">Wednesday</governor>
            <dependent idx="19">early</dependent>
          </dep>
          <dep type="tmod">
            <governor idx="8">hit</governor>
            <dependent idx="20">Wednesday</dependent>
          </dep>
          <dep type="prepc_according_to">
            <governor idx="8">hit</governor>
            <dependent idx="23">to</dependent>
          </dep>
          <dep type="det">
            <governor idx="27">Bureau</governor>
            <dependent idx="24">the</dependent>
          </dep>
          <dep type="nn">
            <governor idx="27">Bureau</governor>
            <dependent idx="25">Central</dependent>
          </dep>
          <dep type="nn">
            <governor idx="27">Bureau</governor>
            <dependent idx="26">Seismology</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="8">hit</governor>
            <dependent idx="27">Bureau</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="8">hit</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="8">hit</governor>
            <dependent idx="1">BEIJING</dependent>
          </dep>
          <dep type="dep">
            <governor idx="8">hit</governor>
            <dependent idx="3">AP</dependent>
          </dep>
          <dep type="det">
            <governor idx="7">earthquake</governor>
            <dependent idx="5">A</dependent>
          </dep>
          <dep type="amod">
            <governor idx="7">earthquake</governor>
            <dependent idx="6">strong</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="8">hit</governor>
            <dependent idx="7">earthquake</dependent>
          </dep>
          <dep type="det">
            <governor idx="11">area</governor>
            <dependent idx="9">a</dependent>
          </dep>
          <dep type="amod">
            <governor idx="11">area</governor>
            <dependent idx="10">remote</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="8">hit</governor>
            <dependent idx="11">area</dependent>
          </dep>
          <dep type="det">
            <governor idx="15">mountains</governor>
            <dependent idx="13">the</dependent>
          </dep>
          <dep type="nn">
            <governor idx="15">mountains</governor>
            <dependent idx="14">Altai</dependent>
          </dep>
          <dep type="prep_in">
            <governor idx="8">hit</governor>
            <dependent idx="15">mountains</dependent>
          </dep>
          <dep type="amod">
            <governor idx="18">Xinjiang</governor>
            <dependent idx="17">northwestern</dependent>
          </dep>
          <dep type="prep_in">
            <governor idx="8">hit</governor>
            <dependent idx="18">Xinjiang</dependent>
          </dep>
          <dep type="amod">
            <governor idx="20">Wednesday</governor>
            <dependent idx="19">early</dependent>
          </dep>
          <dep type="tmod">
            <governor idx="8">hit</governor>
            <dependent idx="20">Wednesday</dependent>
          </dep>
          <dep type="prepc_according_to">
            <governor idx="8">hit</governor>
            <dependent idx="23">to</dependent>
          </dep>
          <dep type="det">
            <governor idx="27">Bureau</governor>
            <dependent idx="24">the</dependent>
          </dep>
          <dep type="nn">
            <governor idx="27">Bureau</governor>
            <dependent idx="25">Central</dependent>
          </dep>
          <dep type="nn">
            <governor idx="27">Bureau</governor>
            <dependent idx="26">Seismology</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="8">hit</governor>
            <dependent idx="27">Bureau</dependent>
          </dep>
        </dependencies>
      </sentence>
      <sentence id="3" sentimentValue="1" sentiment="Negative">
        <tokens>
          <token id="1">
            <word>No</word>
            <lemma>no</lemma>
            <CharacterOffsetBegin>257</CharacterOffsetBegin>
            <CharacterOffsetEnd>259</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="2">
            <word>information</word>
            <lemma>information</lemma>
            <CharacterOffsetBegin>260</CharacterOffsetBegin>
            <CharacterOffsetEnd>271</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="3">
            <word>had</word>
            <lemma>have</lemma>
            <CharacterOffsetBegin>272</CharacterOffsetBegin>
            <CharacterOffsetEnd>275</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="4">
            <word>been</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>276</CharacterOffsetBegin>
            <CharacterOffsetEnd>280</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="5">
            <word>received</word>
            <lemma>receive</lemma>
            <CharacterOffsetBegin>281</CharacterOffsetBegin>
            <CharacterOffsetEnd>289</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="6">
            <word>about</word>
            <lemma>about</lemma>
            <CharacterOffsetBegin>290</CharacterOffsetBegin>
            <CharacterOffsetEnd>295</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="7">
            <word>injuries</word>
            <lemma>injury</lemma>
            <CharacterOffsetBegin>296</CharacterOffsetBegin>
            <CharacterOffsetEnd>304</CharacterOffsetEnd>
            <POS>NNS</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="8">
            <word>or</word>
            <lemma>or</lemma>
            <CharacterOffsetBegin>305</CharacterOffsetBegin>
            <CharacterOffsetEnd>307</CharacterOffsetEnd>
            <POS>CC</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="9">
            <word>damage</word>
            <lemma>damage</lemma>
            <CharacterOffsetBegin>308</CharacterOffsetBegin>
            <CharacterOffsetEnd>314</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="10">
            <word>from</word>
            <lemma>from</lemma>
            <CharacterOffsetBegin>315</CharacterOffsetBegin>
            <CharacterOffsetEnd>319</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="11">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>320</CharacterOffsetBegin>
            <CharacterOffsetEnd>323</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="12">
            <word>magnitude</word>
            <lemma>magnitude</lemma>
            <CharacterOffsetBegin>324</CharacterOffsetBegin>
            <CharacterOffsetEnd>333</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="13">
            <word>6.1</word>
            <lemma>6.1</lemma>
            <CharacterOffsetBegin>334</CharacterOffsetBegin>
            <CharacterOffsetEnd>337</CharacterOffsetEnd>
            <POS>CD</POS>
            <NER>NUMBER</NER>
            <NormalizedNER>6.1</NormalizedNER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="14">
            <word>quake</word>
            <lemma>quake</lemma>
            <CharacterOffsetBegin>338</CharacterOffsetBegin>
            <CharacterOffsetEnd>343</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="15">
            <word>,</word>
            <lemma>,</lemma>
            <CharacterOffsetBegin>343</CharacterOffsetBegin>
            <CharacterOffsetEnd>344</CharacterOffsetEnd>
            <POS>,</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="16">
            <word>which</word>
            <lemma>which</lemma>
            <CharacterOffsetBegin>345</CharacterOffsetBegin>
            <CharacterOffsetEnd>350</CharacterOffsetEnd>
            <POS>WDT</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="17">
            <word>struck</word>
            <lemma>strike</lemma>
            <CharacterOffsetBegin>351</CharacterOffsetBegin>
            <CharacterOffsetEnd>357</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="18">
            <word>at</word>
            <lemma>at</lemma>
            <CharacterOffsetBegin>358</CharacterOffsetBegin>
            <CharacterOffsetEnd>360</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="19">
            <word>2:43</word>
            <lemma>2:43</lemma>
            <CharacterOffsetBegin>361</CharacterOffsetBegin>
            <CharacterOffsetEnd>365</CharacterOffsetEnd>
            <POS>CD</POS>
            <NER>TIME</NER>
            <NormalizedNER>T02:43</NormalizedNER>
            <Speaker>PER0</Speaker>
            <Timex tid="t2" type="TIME">T02:43</Timex>
          </token>
          <token id="20">
            <word>am</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>366</CharacterOffsetBegin>
            <CharacterOffsetEnd>368</CharacterOffsetEnd>
            <POS>VBP</POS>
            <NER>TIME</NER>
            <NormalizedNER>T02:43</NormalizedNER>
            <Speaker>PER0</Speaker>
            <Timex tid="t2" type="TIME">T02:43</Timex>
          </token>
          <token id="21">
            <word>-LRB-</word>
            <lemma>-lrb-</lemma>
            <CharacterOffsetBegin>369</CharacterOffsetBegin>
            <CharacterOffsetEnd>370</CharacterOffsetEnd>
            <POS>-LRB-</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="22">
            <word>1843</word>
            <lemma>1843</lemma>
            <CharacterOffsetBegin>370</CharacterOffsetBegin>
            <CharacterOffsetEnd>374</CharacterOffsetEnd>
            <POS>CD</POS>
            <NER>DATE</NER>
            <NormalizedNER>1843</NormalizedNER>
            <Speaker>PER0</Speaker>
            <Timex tid="t3" type="DATE">1843</Timex>
          </token>
          <token id="23">
            <word>GMT</word>
            <lemma>GMT</lemma>
            <CharacterOffsetBegin>375</CharacterOffsetBegin>
            <CharacterOffsetEnd>378</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>MISC</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="24">
            <word>-RRB-</word>
            <lemma>-rrb-</lemma>
            <CharacterOffsetBegin>378</CharacterOffsetBegin>
            <CharacterOffsetEnd>379</CharacterOffsetEnd>
            <POS>-RRB-</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="25">
            <word>,</word>
            <lemma>,</lemma>
            <CharacterOffsetBegin>379</CharacterOffsetBegin>
            <CharacterOffsetEnd>380</CharacterOffsetEnd>
            <POS>,</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="26">
            <word>said</word>
            <lemma>say</lemma>
            <CharacterOffsetBegin>381</CharacterOffsetBegin>
            <CharacterOffsetEnd>385</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="27">
            <word>a</word>
            <lemma>a</lemma>
            <CharacterOffsetBegin>386</CharacterOffsetBegin>
            <CharacterOffsetEnd>387</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="28">
            <word>bureau</word>
            <lemma>bureau</lemma>
            <CharacterOffsetBegin>388</CharacterOffsetBegin>
            <CharacterOffsetEnd>394</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="29">
            <word>official</word>
            <lemma>official</lemma>
            <CharacterOffsetBegin>395</CharacterOffsetBegin>
            <CharacterOffsetEnd>403</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="30">
            <word>who</word>
            <lemma>who</lemma>
            <CharacterOffsetBegin>404</CharacterOffsetBegin>
            <CharacterOffsetEnd>407</CharacterOffsetEnd>
            <POS>WP</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="31">
            <word>identified</word>
            <lemma>identify</lemma>
            <CharacterOffsetBegin>408</CharacterOffsetBegin>
            <CharacterOffsetEnd>418</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="32">
            <word>himself</word>
            <lemma>himself</lemma>
            <CharacterOffsetBegin>419</CharacterOffsetBegin>
            <CharacterOffsetEnd>426</CharacterOffsetEnd>
            <POS>PRP</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="33">
            <word>only</word>
            <lemma>only</lemma>
            <CharacterOffsetBegin>427</CharacterOffsetBegin>
            <CharacterOffsetEnd>431</CharacterOffsetEnd>
            <POS>RB</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="34">
            <word>by</word>
            <lemma>by</lemma>
            <CharacterOffsetBegin>432</CharacterOffsetBegin>
            <CharacterOffsetEnd>434</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="35">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>435</CharacterOffsetBegin>
            <CharacterOffsetEnd>438</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="36">
            <word>surname</word>
            <lemma>surname</lemma>
            <CharacterOffsetBegin>439</CharacterOffsetBegin>
            <CharacterOffsetEnd>446</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="37">
            <word>Tang</word>
            <lemma>Tang</lemma>
            <CharacterOffsetBegin>447</CharacterOffsetBegin>
            <CharacterOffsetEnd>451</CharacterOffsetEnd>
            <POS>NNP</POS>
            <NER>PERSON</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="38">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>451</CharacterOffsetBegin>
            <CharacterOffsetEnd>452</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
        </tokens>
        <parse>(ROOT (SINV (S (NP (DT No) (NN information)) (VP (VBD had) (VP (VBN been) (VP (VBN received) (PP (IN about) (NP (NP (NNS injuries)) (CC or) (NP (NP (NP (NN damage)) (PP (IN from) (NP (NP (DT the) (NN magnitude) (CD 6.1) (NN quake)) (, ,) (SBAR (WHNP (WDT which)) (S (VP (VBD struck) (PP (IN at) (NP (NP (CD 2:43)) (VP (VBP am)))))))))) (-LRB- -LRB-) (NP (CD 1843) (NNP GMT)) (-RRB- -RRB-)))))))) (, ,) (VP (VBD said) (NP (NP (DT a) (NN bureau) (NN official)) (SBAR (WHNP (WP who)) (S (VP (VBD identified) (NP (PRP himself)) (ADVP (RB only)) (PP (IN by) (NP (DT the) (NN surname)))))))) (NP (NNP Tang)) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="26">said</dependent>
          </dep>
          <dep type="det">
            <governor idx="2">information</governor>
            <dependent idx="1">No</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="5">received</governor>
            <dependent idx="2">information</dependent>
          </dep>
          <dep type="aux">
            <governor idx="5">received</governor>
            <dependent idx="3">had</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="5">received</governor>
            <dependent idx="4">been</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="26">said</governor>
            <dependent idx="5">received</dependent>
          </dep>
          <dep type="prep">
            <governor idx="5">received</governor>
            <dependent idx="6">about</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="6">about</governor>
            <dependent idx="7">injuries</dependent>
          </dep>
          <dep type="cc">
            <governor idx="7">injuries</governor>
            <dependent idx="8">or</dependent>
          </dep>
          <dep type="conj">
            <governor idx="7">injuries</governor>
            <dependent idx="9">damage</dependent>
          </dep>
          <dep type="prep">
            <governor idx="9">damage</governor>
            <dependent idx="10">from</dependent>
          </dep>
          <dep type="det">
            <governor idx="14">quake</governor>
            <dependent idx="11">the</dependent>
          </dep>
          <dep type="nn">
            <governor idx="14">quake</governor>
            <dependent idx="12">magnitude</dependent>
          </dep>
          <dep type="num">
            <governor idx="14">quake</governor>
            <dependent idx="13">6.1</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="10">from</governor>
            <dependent idx="14">quake</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="17">struck</governor>
            <dependent idx="16">which</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="14">quake</governor>
            <dependent idx="17">struck</dependent>
          </dep>
          <dep type="prep">
            <governor idx="17">struck</governor>
            <dependent idx="18">at</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="18">at</governor>
            <dependent idx="19">2:43</dependent>
          </dep>
          <dep type="dep">
            <governor idx="19">2:43</governor>
            <dependent idx="20">am</dependent>
          </dep>
          <dep type="num">
            <governor idx="23">GMT</governor>
            <dependent idx="22">1843</dependent>
          </dep>
          <dep type="dep">
            <governor idx="9">damage</governor>
            <dependent idx="23">GMT</dependent>
          </dep>
          <dep type="det">
            <governor idx="29">official</governor>
            <dependent idx="27">a</dependent>
          </dep>
          <dep type="nn">
            <governor idx="29">official</governor>
            <dependent idx="28">bureau</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="26">said</governor>
            <dependent idx="29">official</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="31">identified</governor>
            <dependent idx="30">who</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="29">official</governor>
            <dependent idx="31">identified</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="31">identified</governor>
            <dependent idx="32">himself</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="31">identified</governor>
            <dependent idx="33">only</dependent>
          </dep>
          <dep type="prep">
            <governor idx="31">identified</governor>
            <dependent idx="34">by</dependent>
          </dep>
          <dep type="det">
            <governor idx="36">surname</governor>
            <dependent idx="35">the</dependent>
          </dep>
          <dep type="pobj">
            <governor idx="34">by</governor>
            <dependent idx="36">surname</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="26">said</governor>
            <dependent idx="37">Tang</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="26">said</dependent>
          </dep>
          <dep type="det">
            <governor idx="2">information</governor>
            <dependent idx="1">No</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="5">received</governor>
            <dependent idx="2">information</dependent>
          </dep>
          <dep type="aux">
            <governor idx="5">received</governor>
            <dependent idx="3">had</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="5">received</governor>
            <dependent idx="4">been</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="26">said</governor>
            <dependent idx="5">received</dependent>
          </dep>
          <dep type="prep_about">
            <governor idx="5">received</governor>
            <dependent idx="7">injuries</dependent>
          </dep>
          <dep type="conj_or">
            <governor idx="7">injuries</governor>
            <dependent idx="9">damage</dependent>
          </dep>
          <dep type="det">
            <governor idx="14">quake</governor>
            <dependent idx="11">the</dependent>
          </dep>
          <dep type="nn">
            <governor idx="14">quake</governor>
            <dependent idx="12">magnitude</dependent>
          </dep>
          <dep type="num">
            <governor idx="14">quake</governor>
            <dependent idx="13">6.1</dependent>
          </dep>
          <dep type="prep_from">
            <governor idx="9">damage</governor>
            <dependent idx="14">quake</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="17">struck</governor>
            <dependent idx="16">which</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="14">quake</governor>
            <dependent idx="17">struck</dependent>
          </dep>
          <dep type="prep_at">
            <governor idx="17">struck</governor>
            <dependent idx="19">2:43</dependent>
          </dep>
          <dep type="dep">
            <governor idx="19">2:43</governor>
            <dependent idx="20">am</dependent>
          </dep>
          <dep type="num">
            <governor idx="23">GMT</governor>
            <dependent idx="22">1843</dependent>
          </dep>
          <dep type="dep">
            <governor idx="9">damage</governor>
            <dependent idx="23">GMT</dependent>
          </dep>
          <dep type="det">
            <governor idx="29">official</governor>
            <dependent idx="27">a</dependent>
          </dep>
          <dep type="nn">
            <governor idx="29">official</governor>
            <dependent idx="28">bureau</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="26">said</governor>
            <dependent idx="29">official</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="31">identified</governor>
            <dependent idx="30">who</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="29">official</governor>
            <dependent idx="31">identified</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="31">identified</governor>
            <dependent idx="32">himself</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="31">identified</governor>
            <dependent idx="33">only</dependent>
          </dep>
          <dep type="det">
            <governor idx="36">surname</governor>
            <dependent idx="35">the</dependent>
          </dep>
          <dep type="prep_by">
            <governor idx="31">identified</governor>
            <dependent idx="36">surname</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="26">said</governor>
            <dependent idx="37">Tang</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="26">said</dependent>
          </dep>
          <dep type="det">
            <governor idx="2">information</governor>
            <dependent idx="1">No</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="5">received</governor>
            <dependent idx="2">information</dependent>
          </dep>
          <dep type="aux">
            <governor idx="5">received</governor>
            <dependent idx="3">had</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="5">received</governor>
            <dependent idx="4">been</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="26">said</governor>
            <dependent idx="5">received</dependent>
          </dep>
          <dep type="prep_about">
            <governor idx="5">received</governor>
            <dependent idx="7">injuries</dependent>
          </dep>
          <dep type="prep_about">
            <governor idx="5">received</governor>
            <dependent idx="9">damage</dependent>
          </dep>
          <dep type="conj_or">
            <governor idx="7">injuries</governor>
            <dependent idx="9">damage</dependent>
          </dep>
          <dep type="det">
            <governor idx="14">quake</governor>
            <dependent idx="11">the</dependent>
          </dep>
          <dep type="nn">
            <governor idx="14">quake</governor>
            <dependent idx="12">magnitude</dependent>
          </dep>
          <dep type="num">
            <governor idx="14">quake</governor>
            <dependent idx="13">6.1</dependent>
          </dep>
          <dep type="prep_from">
            <governor idx="9">damage</governor>
            <dependent idx="14">quake</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="17">struck</governor>
            <dependent idx="16">which</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="14">quake</governor>
            <dependent idx="17">struck</dependent>
          </dep>
          <dep type="prep_at">
            <governor idx="17">struck</governor>
            <dependent idx="19">2:43</dependent>
          </dep>
          <dep type="dep">
            <governor idx="19">2:43</governor>
            <dependent idx="20">am</dependent>
          </dep>
          <dep type="num">
            <governor idx="23">GMT</governor>
            <dependent idx="22">1843</dependent>
          </dep>
          <dep type="dep">
            <governor idx="9">damage</governor>
            <dependent idx="23">GMT</dependent>
          </dep>
          <dep type="det">
            <governor idx="29">official</governor>
            <dependent idx="27">a</dependent>
          </dep>
          <dep type="nn">
            <governor idx="29">official</governor>
            <dependent idx="28">bureau</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="26">said</governor>
            <dependent idx="29">official</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="31">identified</governor>
            <dependent idx="30">who</dependent>
          </dep>
          <dep type="rcmod">
            <governor idx="29">official</governor>
            <dependent idx="31">identified</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="31">identified</governor>
            <dependent idx="32">himself</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="31">identified</governor>
            <dependent idx="33">only</dependent>
          </dep>
          <dep type="det">
            <governor idx="36">surname</governor>
            <dependent idx="35">the</dependent>
          </dep>
          <dep type="prep_by">
            <governor idx="31">identified</governor>
            <dependent idx="36">surname</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="26">said</governor>
            <dependent idx="37">Tang</dependent>
          </dep>
        </dependencies>
      </sentence>
      <sentence id="4" sentimentValue="1" sentiment="Negative">
        <tokens>
          <token id="1">
            <word>He</word>
            <lemma>he</lemma>
            <CharacterOffsetBegin>454</CharacterOffsetBegin>
            <CharacterOffsetEnd>456</CharacterOffsetEnd>
            <POS>PRP</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="2">
            <word>said</word>
            <lemma>say</lemma>
            <CharacterOffsetBegin>457</CharacterOffsetBegin>
            <CharacterOffsetEnd>461</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="3">
            <word>he</word>
            <lemma>he</lemma>
            <CharacterOffsetBegin>462</CharacterOffsetBegin>
            <CharacterOffsetEnd>464</CharacterOffsetEnd>
            <POS>PRP</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="4">
            <word>did</word>
            <lemma>do</lemma>
            <CharacterOffsetBegin>465</CharacterOffsetBegin>
            <CharacterOffsetEnd>468</CharacterOffsetEnd>
            <POS>VBD</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="5">
            <word>not</word>
            <lemma>not</lemma>
            <CharacterOffsetBegin>469</CharacterOffsetBegin>
            <CharacterOffsetEnd>472</CharacterOffsetEnd>
            <POS>RB</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="6">
            <word>expect</word>
            <lemma>expect</lemma>
            <CharacterOffsetBegin>473</CharacterOffsetBegin>
            <CharacterOffsetEnd>479</CharacterOffsetEnd>
            <POS>VB</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="7">
            <word>damages</word>
            <lemma>damages</lemma>
            <CharacterOffsetBegin>480</CharacterOffsetBegin>
            <CharacterOffsetEnd>487</CharacterOffsetEnd>
            <POS>NNS</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="8">
            <word>or</word>
            <lemma>or</lemma>
            <CharacterOffsetBegin>488</CharacterOffsetBegin>
            <CharacterOffsetEnd>490</CharacterOffsetEnd>
            <POS>CC</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="9">
            <word>injuries</word>
            <lemma>injury</lemma>
            <CharacterOffsetBegin>491</CharacterOffsetBegin>
            <CharacterOffsetEnd>499</CharacterOffsetEnd>
            <POS>NNS</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="10">
            <word>to</word>
            <lemma>to</lemma>
            <CharacterOffsetBegin>500</CharacterOffsetBegin>
            <CharacterOffsetEnd>502</CharacterOffsetEnd>
            <POS>TO</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="11">
            <word>be</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>503</CharacterOffsetBegin>
            <CharacterOffsetEnd>505</CharacterOffsetEnd>
            <POS>VB</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="12">
            <word>very</word>
            <lemma>very</lemma>
            <CharacterOffsetBegin>506</CharacterOffsetBegin>
            <CharacterOffsetEnd>510</CharacterOffsetEnd>
            <POS>RB</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="13">
            <word>large</word>
            <lemma>large</lemma>
            <CharacterOffsetBegin>511</CharacterOffsetBegin>
            <CharacterOffsetEnd>516</CharacterOffsetEnd>
            <POS>JJ</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="14">
            <word>because</word>
            <lemma>because</lemma>
            <CharacterOffsetBegin>517</CharacterOffsetBegin>
            <CharacterOffsetEnd>524</CharacterOffsetEnd>
            <POS>IN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="15">
            <word>the</word>
            <lemma>the</lemma>
            <CharacterOffsetBegin>525</CharacterOffsetBegin>
            <CharacterOffsetEnd>528</CharacterOffsetEnd>
            <POS>DT</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="16">
            <word>area</word>
            <lemma>area</lemma>
            <CharacterOffsetBegin>529</CharacterOffsetBegin>
            <CharacterOffsetEnd>533</CharacterOffsetEnd>
            <POS>NN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="17">
            <word>is</word>
            <lemma>be</lemma>
            <CharacterOffsetBegin>534</CharacterOffsetBegin>
            <CharacterOffsetEnd>536</CharacterOffsetEnd>
            <POS>VBZ</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="18">
            <word>sparsely</word>
            <lemma>sparsely</lemma>
            <CharacterOffsetBegin>537</CharacterOffsetBegin>
            <CharacterOffsetEnd>545</CharacterOffsetEnd>
            <POS>RB</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="19">
            <word>inhabited</word>
            <lemma>inhabit</lemma>
            <CharacterOffsetBegin>546</CharacterOffsetBegin>
            <CharacterOffsetEnd>555</CharacterOffsetEnd>
            <POS>VBN</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
          <token id="20">
            <word>.</word>
            <lemma>.</lemma>
            <CharacterOffsetBegin>555</CharacterOffsetBegin>
            <CharacterOffsetEnd>556</CharacterOffsetEnd>
            <POS>.</POS>
            <NER>O</NER>
            <Speaker>PER0</Speaker>
          </token>
        </tokens>
        <parse>(ROOT (S (NP (PRP He)) (VP (VBD said) (SBAR (S (NP (PRP he)) (VP (VBD did) (RB not) (VP (VB expect) (NP (NNS damages) (CC or) (NNS injuries)) (S (VP (TO to) (VP (VB be) (ADJP (RB very) (JJ large)) (SBAR (IN because) (S (NP (DT the) (NN area)) (VP (VBZ is) (ADJP (RB sparsely) (VBN inhabited))))))))))))) (. .))) </parse>
        <dependencies type="basic-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="2">said</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="2">said</governor>
            <dependent idx="1">He</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="6">expect</governor>
            <dependent idx="3">he</dependent>
          </dep>
          <dep type="aux">
            <governor idx="6">expect</governor>
            <dependent idx="4">did</dependent>
          </dep>
          <dep type="neg">
            <governor idx="6">expect</governor>
            <dependent idx="5">not</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="2">said</governor>
            <dependent idx="6">expect</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="6">expect</governor>
            <dependent idx="7">damages</dependent>
          </dep>
          <dep type="cc">
            <governor idx="7">damages</governor>
            <dependent idx="8">or</dependent>
          </dep>
          <dep type="conj">
            <governor idx="7">damages</governor>
            <dependent idx="9">injuries</dependent>
          </dep>
          <dep type="aux">
            <governor idx="13">large</governor>
            <dependent idx="10">to</dependent>
          </dep>
          <dep type="cop">
            <governor idx="13">large</governor>
            <dependent idx="11">be</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="13">large</governor>
            <dependent idx="12">very</dependent>
          </dep>
          <dep type="xcomp">
            <governor idx="6">expect</governor>
            <dependent idx="13">large</dependent>
          </dep>
          <dep type="mark">
            <governor idx="19">inhabited</governor>
            <dependent idx="14">because</dependent>
          </dep>
          <dep type="det">
            <governor idx="16">area</governor>
            <dependent idx="15">the</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="19">inhabited</governor>
            <dependent idx="16">area</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="19">inhabited</governor>
            <dependent idx="17">is</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="19">inhabited</governor>
            <dependent idx="18">sparsely</dependent>
          </dep>
          <dep type="advcl">
            <governor idx="13">large</governor>
            <dependent idx="19">inhabited</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="2">said</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="2">said</governor>
            <dependent idx="1">He</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="6">expect</governor>
            <dependent idx="3">he</dependent>
          </dep>
          <dep type="aux">
            <governor idx="6">expect</governor>
            <dependent idx="4">did</dependent>
          </dep>
          <dep type="neg">
            <governor idx="6">expect</governor>
            <dependent idx="5">not</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="2">said</governor>
            <dependent idx="6">expect</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="6">expect</governor>
            <dependent idx="7">damages</dependent>
          </dep>
          <dep type="conj_or">
            <governor idx="7">damages</governor>
            <dependent idx="9">injuries</dependent>
          </dep>
          <dep type="aux">
            <governor idx="13">large</governor>
            <dependent idx="10">to</dependent>
          </dep>
          <dep type="cop">
            <governor idx="13">large</governor>
            <dependent idx="11">be</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="13">large</governor>
            <dependent idx="12">very</dependent>
          </dep>
          <dep type="xcomp">
            <governor idx="6">expect</governor>
            <dependent idx="13">large</dependent>
          </dep>
          <dep type="mark">
            <governor idx="19">inhabited</governor>
            <dependent idx="14">because</dependent>
          </dep>
          <dep type="det">
            <governor idx="16">area</governor>
            <dependent idx="15">the</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="19">inhabited</governor>
            <dependent idx="16">area</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="19">inhabited</governor>
            <dependent idx="17">is</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="19">inhabited</governor>
            <dependent idx="18">sparsely</dependent>
          </dep>
          <dep type="advcl">
            <governor idx="13">large</governor>
            <dependent idx="19">inhabited</dependent>
          </dep>
        </dependencies>
        <dependencies type="collapsed-ccprocessed-dependencies">
          <dep type="root">
            <governor idx="0">ROOT</governor>
            <dependent idx="2">said</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="2">said</governor>
            <dependent idx="1">He</dependent>
          </dep>
          <dep type="nsubj">
            <governor idx="6">expect</governor>
            <dependent idx="3">he</dependent>
          </dep>
          <dep type="aux">
            <governor idx="6">expect</governor>
            <dependent idx="4">did</dependent>
          </dep>
          <dep type="neg">
            <governor idx="6">expect</governor>
            <dependent idx="5">not</dependent>
          </dep>
          <dep type="ccomp">
            <governor idx="2">said</governor>
            <dependent idx="6">expect</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="6">expect</governor>
            <dependent idx="7">damages</dependent>
          </dep>
          <dep type="dobj">
            <governor idx="6">expect</governor>
            <dependent idx="9">injuries</dependent>
          </dep>
          <dep type="conj_or">
            <governor idx="7">damages</governor>
            <dependent idx="9">injuries</dependent>
          </dep>
          <dep type="aux">
            <governor idx="13">large</governor>
            <dependent idx="10">to</dependent>
          </dep>
          <dep type="cop">
            <governor idx="13">large</governor>
            <dependent idx="11">be</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="13">large</governor>
            <dependent idx="12">very</dependent>
          </dep>
          <dep type="xcomp">
            <governor idx="6">expect</governor>
            <dependent idx="13">large</dependent>
          </dep>
          <dep type="mark">
            <governor idx="19">inhabited</governor>
            <dependent idx="14">because</dependent>
          </dep>
          <dep type="det">
            <governor idx="16">area</governor>
            <dependent idx="15">the</dependent>
          </dep>
          <dep type="nsubjpass">
            <governor idx="19">inhabited</governor>
            <dependent idx="16">area</dependent>
          </dep>
          <dep type="auxpass">
            <governor idx="19">inhabited</governor>
            <dependent idx="17">is</dependent>
          </dep>
          <dep type="advmod">
            <governor idx="19">inhabited</governor>
            <dependent idx="18">sparsely</dependent>
          </dep>
          <dep type="advcl">
            <governor idx="13">large</governor>
            <dependent idx="19">inhabited</dependent>
          </dep>
        </dependencies>
      </sentence>
    </sentences>
    <coreference>
      <coreference>
        <mention representative="true">
          <sentence>1</sentence>
          <start>5</start>
          <end>6</end>
          <head>5</head>
          <text>|</text>
        </mention>
        <mention>
          <sentence>1</sentence>
          <start>7</start>
          <end>8</end>
          <head>7</head>
          <text>|</text>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>1</sentence>
          <start>3</start>
          <end>13</end>
          <head>4</head>
          <text>1st Ld-Writethru | URGENT | Earthquake Rocks Northwestern Xinjiang Mountains</text>
        </mention>
        <mention>
          <sentence>1</sentence>
          <start>1</start>
          <end>2</end>
          <head>1</head>
          <text>BC-China-Earthquake</text>
        </mention>
        <mention>
          <sentence>3</sentence>
          <start>32</start>
          <end>33</end>
          <head>32</head>
          <text>himself</text>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>1</sentence>
          <start>10</start>
          <end>13</end>
          <head>12</head>
          <text>Northwestern Xinjiang Mountains</text>
        </mention>
        <mention>
          <sentence>2</sentence>
          <start>17</start>
          <end>19</end>
          <head>18</head>
          <text>northwestern Xinjiang</text>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>2</sentence>
          <start>9</start>
          <end>12</end>
          <head>11</head>
          <text>a remote area</text>
        </mention>
        <mention>
          <sentence>4</sentence>
          <start>15</start>
          <end>17</end>
          <head>16</head>
          <text>the area</text>
        </mention>
      </coreference>
      <coreference>
        <mention representative="true">
          <sentence>3</sentence>
          <start>27</start>
          <end>37</end>
          <head>29</head>
          <text>a bureau official who identified himself only by the surname</text>
        </mention>
        <mention>
          <sentence>4</sentence>
          <start>1</start>
          <end>2</end>
          <head>1</head>
          <text>He</text>
        </mention>
        <mention>
          <sentence>4</sentence>
          <start>3</start>
          <end>4</end>
          <head>3</head>
          <text>he</text>
        </mention>
      </coreference>
    </coreference>
  </document>
</root>   
"""
        f = StringIO(xml_str)
        cls.doc = corenlp.Document(f)


    def gold_transition_bigram_test(cls):
        correct_trans = (Transition(('s-0', 'START'), 0), 
                         Transition(('s-1', 's-0'), 1), 
                         Transition(('s-2', 's-1'), 2), 
                         Transition(('s-3', 's-2'), 3), 
                         Transition(('END', 's-3'), 4)) 

       
        inst = NGramDiscourseInstance(cls.doc, {}, None, 2)
        gold_trans = tuple(inst.gold_transitions())

        assert correct_trans == gold_trans

    def gold_transition_trigram_test(cls):
        correct_trans = (Transition(('s-0', 'START', 'START'), 0), 
                         Transition(('s-1', 's-0', 'START'), 1), 
                         Transition(('s-2', 's-1', 's-0'), 2), 
                         Transition(('s-3', 's-2', 's-1'), 3), 
                         Transition(('END', 's-3', 's-2'), 4)) 

       
        inst = NGramDiscourseInstance(cls.doc, {}, None, 3)
        gold_trans = tuple(inst.gold_transitions())

        assert correct_trans == gold_trans

    def debug_feature_bigram_test(cls):
        inst = NGramDiscourseInstance(cls.doc, {}, None, 2)
        fmap1 = {}
        t1 = Transition(('s-3', 's-2'), 3)
        inst._f_debug(fmap1, t1)

        assert fmap1.get('DEBUG', 0) == 1

        fmap2 = {}
        t2 = Transition(('s-3', 's-1'), 3)
        inst._f_debug(fmap2, t2)
        
        assert fmap2.get('DEBUG', 0) == 0

        fmap3 = {}
        t3 = Transition(('s-0', 'START'), 0)
        inst._f_debug(fmap3, t3)
        
        assert fmap3.get('DEBUG', 0) == 1

        fmap4 = {}
        t4 = Transition(('s-3', 'START'), 0)
        inst._f_debug(fmap4, t4)
        
        assert fmap4.get('DEBUG', 0) == 0

    def debug_feature_trigram_test(cls):
        inst = NGramDiscourseInstance(cls.doc, {}, None, 3)
        fmap1 = {}
        t1 = Transition(('s-3', 's-2', 's-1'), 3)
        inst._f_debug(fmap1, t1)

        assert fmap1.get('DEBUG', 0) == 1

        fmap2 = {}
        t2 = Transition(('s-3', 's-1', 's-2'), 3)
        inst._f_debug(fmap2, t2)
        
        assert fmap2.get('DEBUG', 0) == 0

        fmap3 = {}
        t3 = Transition(('s-0', 'START', 'START'), 0)
        inst._f_debug(fmap3, t3)
        
        assert fmap3.get('DEBUG', 0) == 1

        fmap4 = {}
        t4 = Transition(('s-3', 'START', 'START'), 0)
        inst._f_debug(fmap4, t4)
        
        assert fmap4.get('DEBUG', 0) == 0



