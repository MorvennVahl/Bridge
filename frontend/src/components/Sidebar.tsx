import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Workflow" },
  { to: "/predict", label: "Predict" },
  { to: "/compute", label: "Compute" },
  { to: "/data", label: "Data" },
];

export default function Sidebar() {
  return (
    <aside className="flex h-full w-56 shrink-0 flex-col border-r border-slate-200 bg-white">
      <div className="border-b border-slate-200 px-5 py-5">
        <div className="text-base font-semibold tracking-tight text-slate-900">Bridge</div>
        <div className="mt-0.5 text-[11px] text-slate-500">Frank · Adam · Alex</div>
      </div>
      <nav className="flex-1 px-3 py-4">
        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.to === "/"}
            className={({ isActive }) =>
              `mb-1 block rounded-md px-3 py-2 text-sm font-medium transition ${
                isActive
                  ? "bg-slate-900 text-white"
                  : "text-slate-600 hover:bg-slate-100 hover:text-slate-900"
              }`
            }
          >
            {l.label}
          </NavLink>
        ))}
      </nav>
      <div className="border-t border-slate-200 px-5 py-3 text-[11px] text-slate-400">
        Modal × Claude Science
      </div>
    </aside>
  );
}
