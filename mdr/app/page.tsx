import MagicSquareVisualizer from './MagicSquareVisualizer'

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1 className="text-3xl font-bold mb-8">Macrodata Refinement</h1>
      <MagicSquareVisualizer />
    </main>
  )
}