import java.util.ArrayList;
import java.util.Random;

import weka.clusterers.Clusterer;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

class Row {
	
	private int id;
	private Instance x;
	private int	id_cluster;
	
	Row(int id, Instance x, int id_cluster)
	{
		this.id = id;
		this.x = x;
		this.id_cluster = id_cluster;
	}
	
	public int getId()
	{
		return id;
	}
	
	public Instance getInstance()
	{
		return x;
	}
	
	public int getCluster()
	{
		return id_cluster;
	}
	
	public void setCluster(int id_cluster)
	{
		this.id_cluster = id_cluster;
	}
}

class Cluster {
	
	int id;
	
	double[] center;
	
	Cluster(int id)
	{
		this.id = id;
	}
	
	public void adjustCenter(ArrayList<Row> table)
	{	
		/*
		 * Untuk setiap instance yang punya cluster = id
		 * dicari centernya dengan cara:
		 * 
		 * 
		 */
		
		int numAttributes = table.get(0).getInstance().numAttributes();
		center = new double[numAttributes];
		
		double[] temp = new double[numAttributes];
		int numMembers = 0;		
		
		for(int i=0; i < table.size(); i++)
		{
		
			if(table.get(i).getCluster() == id)
			{
				for(int j = 0; j < numAttributes; j++)
				{
					temp[j] += table.get(i).getInstance().value(j);
				}
				
				numMembers++;
			}
			
		}
		
		for(int k=0; k < numAttributes; k++)
		{
			center[k] = Math.ceil(temp[k] / numMembers);
			System.out.print(center[k] + " | ");
		}
		
		System.out.println();
		
	}
	
	public double getDistanceToCenter(Instance x)
	{
		/*
		 * distance yang diambil sigma((xi-c)^2) --> gak pake kuadrat
		 */
		
		double sum = 0;
		
		for(int i = 0; i < x.numAttributes(); i++)
		{
			sum += Math.pow(x.value(i) - center[i], 2);
		}
		
		return sum;
	}
	
}

public class KMeans implements Clusterer{
	
	private int numCluster = 2;
	private int maxEpoch = 2;
	private int currEpoch = 0;
	
	private ArrayList<Row> table;
	private ArrayList<Cluster> clusters;
	private ArrayList<Row> prevTable;
	
	KMeans(int numCluster)
	{
		this.numCluster = numCluster;
		table = new ArrayList<>();
		initCluster(numCluster);
	}
	
	KMeans(int numCluster, int maxEpoch)
	{
		this.numCluster = numCluster;
		this.maxEpoch = maxEpoch;
		table = new ArrayList<>();
		initCluster(numCluster);
	}
	
	private void initCluster(int numCluster)
	{
		clusters = new ArrayList<>();
		for(int i=0; i < numCluster; i++)
		{
			clusters.add(new Cluster(i+1));
		}
	}
	
	@Override
	public void buildClusterer(Instances dataset) throws Exception {
		
		/*
		 * Setiap instance dimasukkin dulu ke table
		 * dan diassign masuk ke cluster mana
		 */
		
		for(int i = 0; i < dataset.numInstances(); i++)
		{
			int randomCluster = new Random().nextInt(this.numCluster) + 1;
			Row row = new Row(i+1, dataset.instance(i), randomCluster);
			table.add(row);
		}
		
		System.out.println("[EPOCH: Inisialisasi]");
		System.out.println();
		printTable();

		/*
		 * Sekarang hitung center untuk masing-masing
		 * cluster
		 */
		System.out.println("[CENTER DATA]");
		for(int i = 0; i < clusters.size(); i++)
		{
			clusters.get(i).adjustCenter(table);
		}
				
		while((currEpoch < maxEpoch) && !isConverge())
		{
			
			System.out.println();
			
			System.out.println("[EPOCH: " + currEpoch + "]");
			
			/*
			 * Save to previous table
			 */
			prevTable = table;
			
			/*
			 * Sekarang untuk tiap-tiap instance dihitung
			 * jaraknya ke masing-masing center dari cluster
			 * lalu assign instance tersebut ke yang paling
			 * dekat jaraknya 
			 */
			for(int i = 0; i < table.size(); i++)
			{
				int min_id_cluster = 0;
				double tmp_distance = Double.MAX_VALUE;
				
				for(int j = 0; j < clusters.size(); j++)
				{
					double distance = clusters.get(j).getDistanceToCenter(table.get(i).getInstance());
					//System.out.println("distance: " + distance);
					if(distance < tmp_distance)
					{
						tmp_distance = distance;
						min_id_cluster = j+1;
					}
				}
				
				//System.out.println(min_id_cluster);
				table.get(i).setCluster(min_id_cluster);
				//System.out.println();
			}
			
			System.out.println();
			printTable();
			
			/*
			 * Sekarang hitung lagi center untuk 
			 * masing-masing cluster
			 */
			System.out.println("[CENTER DATA]");
			for(int i = 0; i < clusters.size(); i++)
			{
				clusters.get(i).adjustCenter(table);
			}
			
			currEpoch++;
		}
		
		System.out.println();
		System.out.println("[DATA SUDAH KONVERGEN ATAU MENCAPAI MAX EPOCH]");
		System.out.println();
		printTable();
		
	}
	
	public boolean isConverge()
	{
		boolean ret = true;
		boolean foundMistake = false;
		int i = 0;
		
		if(prevTable != null)
		{
			while(!foundMistake && (i < table.size()))
			{
				if(table.get(i).getCluster() != prevTable.get(i).getCluster())
				{
					foundMistake = true;
				}
				else
				{
					i++;
				}
			}
		} else
		{
			foundMistake = true;
		}
		
		if(foundMistake)
		{
			ret = false;
		}
		
		return ret;
	}
	
	public void printTable()
	{
		System.out.println("[TABEL DATA]");
		
		for(int i = 0; i < table.size(); i++)
		{
			Row currRow = table.get(i);
			System.out.println(currRow.getInstance() + " | " + currRow.getCluster());
		}
		
		System.out.println();
	}

	@Override
	public int clusterInstance(Instance x) throws Exception {
		
		/*
		 * Instance dihitung jaraknya ke masing-masing 
		 * center dari cluster lalu tentukan instance 
		 * di assign kemana
		 */
		int min_id_cluster = 0;
		double tmp_distance = Double.MAX_VALUE;
		
		for(int j = 0; j < clusters.size(); j++)
		{
			double distance = clusters.get(j).getDistanceToCenter(x);
			if(distance < tmp_distance)
			{
				tmp_distance = distance;
				min_id_cluster = j+1;
			}
		}
		
		System.out.println(x + " | " + min_id_cluster);
				
		return 0;
		
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int numberOfClusters() throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}
	
	public static void main(String[] args) {
		try {
			DataSource source = new DataSource("weather.nominal.arff");
			Instances data = source.getDataSet();
			
			if(data.classIndex() == -1)
				data.setClassIndex(data.numAttributes() - 1);
			
			KMeans clusterer = new KMeans(2, 1000);
			clusterer.buildClusterer(data);
			
			// Setelah clusterer-nya dibuild baru test data
			DataSource test_source = new DataSource("test.arff");
			Instances datatest = test_source.getDataSet();
			
			if(datatest.classIndex() == -1)
				datatest.setClassIndex(datatest.numAttributes() - 1);
			
			System.out.println("[CLUSTERING INSTANCES BARU]");
			for(int i = 0; i < datatest.numInstances(); i++)
			{
				clusterer.clusterInstance(datatest.instance(i));
			}
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
