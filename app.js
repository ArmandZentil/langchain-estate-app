import { OpenAI } from "langchain/llms/openai";
import 'dotenv/config';
// import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
// import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const model = new OpenAI({});


const loader = new TextLoader("dataPdfs/gzWill.txt");

// const loader = new PDFLoader("dataPdfs/gzWill.pdf", {
//   splitPages: false,
// });

// const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
// const text = ;
const docs = await loader.load()

// await textSplitter.createDocuments([text.toString]);

const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

const apiKey = process.env.OPENAI_API_KEY;

const query = process.argv[2];

const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
  const res = await chain.call({
    query: "Who are the trustees?",
  });
  console.log({ res });