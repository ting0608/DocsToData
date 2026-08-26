// Backend API base URL.
//
// English: Empty string means "same origin as the frontend" — correct for
// local dev, Docker, and Cloud Run, where FastAPI serves both the API and
// the static frontend files from one process. Once the frontend is hosted
// separately on Amplify and the backend lives behind API Gateway, this must
// point at the API Gateway invoke URL instead. `amplify.yml` overwrites this
// file's value at build time via the `API_BASE_URL` Amplify environment
// variable (see amplify.yml), so this default only applies when running the
// frontend directly against a local/same-origin backend.
// 中文: 空字串代表「與前端同一個 origin」——這在本機開發、Docker、Cloud Run
// 都是對的，因為 FastAPI 在同一個process同時提供 API 與前端靜態檔案。一旦前端
// 改為獨立部署在 Amplify、後端則在 API Gateway 之後，這裡就必須改成指向
// API Gateway 的 invoke URL。`amplify.yml` 會在建置階段透過 Amplify 環境變數
// `API_BASE_URL` 覆寫這個檔案的值（見 amplify.yml），因此這裡的預設值只適用
// 於前端直接對接本機/同源後端的情況。
export const API_BASE_URL = "";
