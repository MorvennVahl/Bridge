import { Route, Routes } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import ComputePage from "./pages/ComputePage";
import DataPage from "./pages/DataPage";
import PredictPage from "./pages/PredictPage";
import WorkflowPage from "./pages/WorkflowPage";

export default function App() {
  return (
    <div className="flex h-screen bg-slate-50 font-sans text-slate-900">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <Routes>
          <Route path="/" element={<WorkflowPage />} />
          <Route path="/predict" element={<PredictPage />} />
          <Route path="/compute" element={<ComputePage />} />
          <Route path="/data" element={<DataPage />} />
        </Routes>
      </main>
    </div>
  );
}
