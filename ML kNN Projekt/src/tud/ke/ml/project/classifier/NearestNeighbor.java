package tud.ke.ml.project.classifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import tud.ke.ml.project.framework.classifier.ANearestNeighbor;
import tud.ke.ml.project.util.Pair;

/**
 * This implementation assumes the class attribute is always available (but probably not set)
 * @author cwirth
 *
 */
public class NearestNeighbor extends ANearestNeighbor {
	
	protected double[] scaling;
	protected double[] translation;
	
	private List<List<Object>> trainData;
	private HashMap<Object, Integer> apriori;
	
	/**
	 * Compute an a priori indicator of the classes distribution over the training set : 'apriori' attribute.
	 * In case of ex-aequo winners in getWinner(), the winner should be the class which 
	 * has the higher associated number in 'apriori' ie the higher occurrence in training set
	 */
	private void compute_apriori() {
		HashMap<Object, Integer> map = new HashMap<Object, Integer>();
		
		for (List<Object> trainingInstance : trainData) {
			Object instance_class = trainingInstance.get(getClassAttribute());
			if (!map.containsKey(instance_class)) {
				map.put(instance_class, new Integer(0));
			}
			map.put(instance_class, map.get(instance_class)+1);
		}	
		apriori = map;
	}

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> votes;
		if (isInverseWeighting()) {
			votes = getWeightedVotes(subset);
		} else {
			votes = getUnweightedVotes(subset);
		}
		return getWinner(votes);
	}
	
	@Override
	protected void learnModel(List<List<Object>> traindata) {
		// Instance based classifier => there is no learning, we just save all instances of the train data 
		this.trainData = traindata;
		compute_apriori();
	}
	
	@Override
	protected Map<Object, Double> getUnweightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		HashMap<Object, Double> map = new HashMap<Object, Double>();
		
		for (Pair<List<Object>, Double> pair : subset) {
			Object instance_class = pair.getA().get(getClassAttribute());
			if (!map.containsKey(instance_class)) {
				map.put(instance_class, new Double(0));
			}
			map.put(instance_class, map.get(instance_class)+1); // TO TEST
		}	
		return map;
	}
	
	@Override
	protected Map<Object, Double> getWeightedVotes(
			List<Pair<List<Object>, Double>> subset) {
		// die Stimmen als Summe der inversen Distanzen berechnen
		HashMap<Object, Double> map = new HashMap<Object, Double>();
		
		for (Pair<List<Object>, Double> pair : subset) {
			Object instance_class = pair.getA().get(getClassAttribute());
			if (!map.containsKey(instance_class)) {
				map.put(instance_class, new Double(0));
			}
			// TODO Change " == 0" into " < eps"
			Double inverseDistance = (pair.getB() == 0) ? Double.MAX_VALUE : (1/pair.getB()); 
			map.put(instance_class, map.get(instance_class)+inverseDistance);
		}	
		return map;
	}
	
	@Override
	protected Object getWinner(Map<Object, Double> votesFor) {
		// What to do when there are ex-aequo winners ?? 
		// Method : use a apriori fonction so that the winner be the candidate
		// which is the most frequent in training set 
		
		// find the highest Double value
		if ( (votesFor == null) || (votesFor.size() == 0) ) {
			return null;
		}
		Object winner = null;
		List<Object> ex_aequo = new ArrayList<Object>();
		for (Entry<Object, Double> e : votesFor.entrySet()) {
			if (winner == null) {
				winner = e.getKey();
			} else if ( (e.getValue() == votesFor.get(winner)) && (apriori.get(e.getValue()) > apriori.get(winner))) {
				winner = e.getKey();
				ex_aequo.add(e.getKey());
			} else if (e.getValue() > votesFor.get(winner)) {
				winner = e.getKey();
				ex_aequo.clear();
			} 
			
			// Old version
//			if ( (winner == null) || (e.getValue() > votesFor.get(winner)) ) {
//				winner = e.getKey();
//			} 
			
		}
		if (ex_aequo.size() > 0) {
			System.out.println("Ex-aequo : "+ex_aequo.size()+1);
		}
		return winner;
	}
	
	@Override
	protected List<Pair<List<Object>, Double>> getNearest(List<Object> testdata) {
		int k = getkNearest();
		
		Object data[] = new Object[trainData.size()];

		// Initialize scaling and translation
		double[][] t = normalizationScaling();
		scaling = (t==null)? null : t[0];
		translation = (t==null)? null : t[1];
		
		// compute distance from testdata instance to each instance of the trainData
		List<Double> distances = new ArrayList<Double>();
		switch (getMetric()) {
		case 0 :
			// use Manhattan distance
			for (int i=0; i<data.length; i++)  {
				data[i] = new Pair<List<Object>, Double>(trainData.get(i), determineManhattanDistance(testdata, trainData.get(i)));
			}
			break;
		case 1 : 
			// use Euclidian distance
			for (int i=0; i<data.length; i++)  {
				data[i] = new Pair<List<Object>, Double>(trainData.get(i), determineEuclideanDistance(testdata, trainData.get(i)));
			}
			break;
		default :
			throw new RuntimeException("Unhandled metric : "+getMetric());
			
		}
			
		// pick the getKNearest() better instances : with smallest distances
		Arrays.sort(data, new Comparator<Object>() {
			@Override
			public int compare(Object o1, Object o2) {
				Pair<List<Object>, Double> instance1 = (Pair<List<Object>, Double>) o1;
				Pair<List<Object>, Double> instance2 = (Pair<List<Object>, Double>) o2;
				return instance1.getB().compareTo(instance2.getB());
			}
		});
		
		// get the instances whose distance is among the k smallest values 
		// (may be more than k if many instances have the same distance)
		// TODO implement the decision function : see compute_apriori 
		System.out.println("k = "+k); 
		List<Pair<List<Object>, Double>> kNeighbors = new ArrayList<Pair<List<Object>, Double>>();
		for (int i=0; i<k && i<data.length; i++) {
			kNeighbors.add((Pair<List<Object>, Double>)data[i]);
			System.out.println("result["+i+"] = class:"+((Pair<List<Object>, Double>)data[i]).getA().get(getClassAttribute())+" / distance:"+((Pair<List<Object>, Double>)data[i]).getB());
		}
		int lastNeighbor = kNeighbors.size()-1;
		
		while( (lastNeighbor<data.length-1) && (kNeighbors.get(lastNeighbor).getB() == ((Pair<List<Object>, Double>)data[lastNeighbor+1]).getB()) ) {
			lastNeighbor += 1; 
			kNeighbors.add((Pair<List<Object>, Double>)data[lastNeighbor]);
			System.out.println("++ result["+lastNeighbor+"] = class:"+((Pair<List<Object>, Double>)data[lastNeighbor]).getA().get(getClassAttribute())+" / distance:"+((Pair<List<Object>, Double>)data[lastNeighbor]).getB());
		}
		
		return kNeighbors;
	}
	
	/**
	 * normalize the given attribute value into [0,1] with given translation and scaling
	 */
	private double normalize(double attrValue, double trans, double scale) {
		return (attrValue+trans)/scale;
	}
	
	@Override
	protected double determineManhattanDistance(List<Object> instance1,
			List<Object> instance2) {
		
		int nbAttributes = instance1.size();
		double d = 0;
		for (int i=0; i<nbAttributes; i++) {
			if ( (instance1.get(i) instanceof Double ) && (instance2.get(i) instanceof Double ) ) {
				double value1 = (Double) instance1.get(i);
				double value2 = (Double) instance2.get(i);				
				d += ( Math.abs(normalize(value1,translation[i],scaling[i]) - normalize(value2,translation[i],scaling[i]))  );
			} else if ( (instance1.get(i) instanceof String ) && (instance2.get(i) instanceof String ) ) {
				// translation[i] == 0 , scaling[i] == 1
				d += ((String)instance1.get(i)).equals((String)instance2.get(i)) ? 0 : 1;
			} else {
				// Should not happen
				throw new RuntimeException("Attributes don't pass with each other");
			}
		}
		return d;
	}
	
	@Override
	protected double determineEuclideanDistance(List<Object> instance1,
			List<Object> instance2) {
		int nbAttributes = instance1.size();
		double sd2 = 0;
		for (int i=0; i<nbAttributes; i++) {
			if ( (instance1.get(i) instanceof Double ) && (instance2.get(i) instanceof Double ) ) {
				double value1 = (Double) instance1.get(i);
				double value2 = (Double) instance2.get(i);				
				double d = ( Math.abs(normalize(value1,translation[i],scaling[i]) - normalize(value2,translation[i],scaling[i]))  );
				sd2 += Math.pow( d, 2);
			} else if ( (instance1.get(i) instanceof String ) && (instance2.get(i) instanceof String ) ) {
				// translation[i] == 0 , scaling[i] == 1
				double d = ((String)instance1.get(i)).equals((String)instance2.get(i)) ? 0 : 1 ;
				sd2 += Math.pow( d, 2);  //the square of 0 or 1 is still 0 or 1..
			} else {
				// Should not happen
				throw new RuntimeException("Attributes don't pass with each other");
			}
		}
		return Math.sqrt(sd2);
	}
	
	@Override
	protected double[][] normalizationScaling() {
		double[][] results = null;
		double[] maxvalues = null;
		double[] minvalues = null;
		if ( !(trainData == null) && !trainData.isEmpty() ) {
			// dimension of data ie number of attributes 
			int dim = trainData.get(0).size(); 

			results = new double[2][dim];
				
			if (isNormalizing()) {
				maxvalues = new double[dim];
				minvalues = new double[dim];
				Arrays.fill(minvalues, 0);
				Arrays.fill(maxvalues, 1);
				//determine minimal and maximal values per attribute
				for (int inst=0; inst<trainData.size(); inst++) { //iterate over instances
					List<Object> instance;
					instance = trainData.get(inst);
					for (int attr=0; attr<dim; attr++) {	//iterate over attributes
						if (!(instance.get(attr) instanceof String)) { //only consider numerical attributes
							double value = (Double) instance.get(attr);
							if ((minvalues[attr] == 0) && (maxvalues[attr] == 1)) { //initial values
								minvalues[attr] = value;
								maxvalues[attr] = value;
							} else if (minvalues[attr] > value) {	//new minimum
								minvalues[attr] = value;
							} else if (maxvalues[attr] < value) {	//new maximum
								maxvalues[attr] = value;
							}
						}
					}
				}
				//compute scaling and translation
				for (int attr=0; attr<dim; attr++) {
					results[0][attr] = maxvalues[attr] - minvalues[attr]; //1-0=1 for string attributes
					if (results[0][attr] == 0) {	// if all instances have the same value
						results[0][attr] = 1;
					}
					results[1][attr] = -minvalues[attr];	//0 for string attributes
				}
			} else {
				Arrays.fill(results[0], 1);
				Arrays.fill(results[1], 0);
			}
		}
		
		return results;
		
	}
		
	/* Version Soso
	protected double[][] normalizationScaling() {
		double[][] results = null;
		double[] maxvalues = null;
		double[] minvalues = null;
		if ( !(trainData == null) && !trainData.isEmpty() ) {
			// dimension of data ie number of attributes 
			int dim = trainData.get(0).size(); 
			results = new double[dim][2];
			
			// translation
			results[1] = new double[dim];
			if (isNormalizing()) {
				// TODO
				// Look for negative values for each Double attribute in training data
				for (int j=0; j<results.length; j++) {
					for (int i=0; i<trainData.size(); i++) {
						if (trainData.get(i).get(j) instanceof Double) {
							Double d = (Double) trainData.get(i).get(j);
							if ((d<0) && (results[1][j] < -d)) {
								results[1][j] = -d;
							}
						}
					}
				}
			} // else all values are equals to 0
			
			// scaling
			results[0] = new double[dim];
			if (isNormalizing()) {
				// TODO
				// Find the max of each Double attribute in training data
				for (int j=0; j<results.length; j++) {
					for (int i=0; i<trainData.size(); i++) {
						if (trainData.get(i).get(j) instanceof Double) {
							Double d = (Double) trainData.get(i).get(j);
							if (results[0][j] < d + results[1][j]) {
								results[0][j] = d + results[1][j];
							}
						}
					}
				}
			} 
			for (int j=0; j<results[0].length; j++) {
					if (results[0][j] < 1) {
						results[0][j] = 1d;
			}
			}
		}
		
		return results;
			
	}
	*/
	
	@Override
	protected String[] getMatrikelNumbers() {
		// OK
		String[] numbers = new String[3];
		numbers[0] = "1682731"; // Jan
		numbers[1] = "2805397"; // Killian
		numbers[2] = "2939001"; // Solene
		return numbers;
	}

}
