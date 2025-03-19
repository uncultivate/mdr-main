import MagicSquareVisualizer from '@/components/MagicSquareVisualizer';

export default function Home() {
  return (
    <main className="min-h-screen p-8 bg-[#001525]">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold mb-8 text-[#24e4f2]">Macrodata Refinement</h1>
        <p className="text-[#24e4f2] mb-8">Data Engineers, welcome to the Macrodata Refinement challenge!
          <br>Play as a human and use arrow keys to navigate the grid and find magic squares.</br>
          <br>Or add your own custom strategies and see how your AI performs.</br>
        </p>
        <MagicSquareVisualizer />
      </div>
    </main>
  );
}