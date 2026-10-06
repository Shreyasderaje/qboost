import { Routes, Route, useLocation } from "react-router-dom";
import { useEffect } from "react";
import { NavBar, Footer } from "./components/NavBar.jsx";
import Home from "./pages/Home.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Diagnose from "./pages/Diagnose.jsx";
import Network from "./pages/Network.jsx";
import Research from "./pages/Research.jsx";

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

export default function App() {
  return (
    <>
      <ScrollToTop />
      <NavBar />
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/diagnose" element={<Diagnose />} />
          <Route path="/network" element={<Network />} />
          <Route path="/research" element={<Research />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </main>
      <Footer />
    </>
  );
}
